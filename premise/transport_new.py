"""
transport_new.py contains the class Transport, which imports inventories
for different modes and types of transport, updates efficiencies, and creates
fleet average vehicle inventories based on IAM data. The inventories are 
integrated afterwards into the database.   
"""

import copy
import json
import numpy as np
import uuid
import xarray as xr
import yaml
from typing import Dict, List

from .filesystem_constants import DATA_DIR
from .logger import create_logger
from .transformation import BaseTransformation, IAMDataCollection
from .utils import rescale_exchanges
from .validation import TransportValidationNEW
from wurst import searching as ws
from wurst.errors import NoResults

FILEPATH_VEHICLES_MAP = DATA_DIR / "transport" / "vehicles_map_NEW.yaml"

logger = create_logger("transport")
# TODO: work on logger (reporting.yaml & premise_transport.log)
# TODO: work on change report
# TODO: work on scenario report

def _update_transport(scenario, version, system_model):
    transport = Transport(
        database=scenario["database"],
        year=scenario["year"],
        model=scenario["model"],
        pathway=scenario["pathway"],
        iam_data=scenario["iam data"],
        version=version,
        system_model=system_model,
        index=scenario.get("index"),
    )
    
    if scenario["iam data"].roadfreight_markets is not None and scenario["iam data"].railfreight_markets is not None:
        transport.generate_datasets()
        transport.relink_datasets()
        transport.generate_transport_markets()
        transport.generate_unspecified_transport_vehicles()
        transport.empty_ecoinvent_datasets()
        # transport.relink_exchanges() #TODO: there are relinks that link to non exisitng datasets (these do not exist because the IAM does not provide varibles for them)
        scenario["database"] = transport.database 
        scenario["cache"] = transport.cache
        scenario["index"] = transport.index
        
        validate = TransportValidationNEW(
            model=scenario["model"],
            scenario=scenario["pathway"],
            year=scenario["year"],
            regions=scenario["iam data"].regions,
            database=transport.database,
            iam_data=scenario["iam data"],
        )
        
        validate.run_transport_checks()
    
    elif scenario["iam data"].roadfreight_markets is not None and scenario["iam data"].railfreight_markets is None:
        print("No railfreight markets found in IAM data. Skipping freight transport.")
        
    elif scenario["iam data"].roadfreight_markets is None and scenario["iam data"].railfreight_markets is not None:
        print("No roadfreight markets found in IAM data. Skipping freight transport.")
        
    else:
        print("No transport markets found in IAM data. Skipping freight transport.")
        
    # TODO: if one transport market is not found the other one will still be updateable?
    
    return scenario


def get_vehicles_mapping() -> Dict[str, dict]:
    """
    Return a dictionary that contains mapping
    between `ecoinvent` terminology and `premise` terminology
    regarding size classes, powertrain types, etc.
    
    :return: dictionary to map terminology between carculator and ecoinvent
    
    """
    with open(FILEPATH_VEHICLES_MAP, "r", encoding="utf-8") as stream:
        out = yaml.safe_load(stream)
        return out


def normalize_exchange_amounts(list_act: List[dict]) -> List[dict]:
    """
    In vehicle market datasets, we need to ensure that the total contribution
    of single vehicle types equal 1.

    :param list_act: list of transport market activities
    :return: same list, with activity exchanges normalized to 1

    """

    for act in list_act:
        total = 0
        for exc in act["exchanges"]:
            if exc["type"] == "technosphere":
                total += exc["amount"]

        for exc in act["exchanges"]:
            if exc["type"] == "technosphere":
                exc["amount"] /= total

    return list_act


class Transport(BaseTransformation):
    """
    Class that modifies transport inventory datasets based on IAM data.
    It stores functions to generate transport datasets for all IAM regions,
    based on newly imported LCIs, incl. adjusting their efficiencies for a given year.
    It creates market processes for freight transport and relinks exchanges of
    datasets using transport inventories. It deletes the old ecoinvent transport datasets.    
    """
    
    def __init__(
        self,
        database: List[dict],
        iam_data: IAMDataCollection,
        model: str,
        pathway: str,
        year: int,
        version: str,
        system_model: str,
        index: dict = None,
    ):
        super().__init__(
            database,
            iam_data,
            model,
            pathway,
            year,
            version,
            system_model,
            index,
        )
    
    def generate_datasets(self):
        """
        Function that creates inventories for IAM region based on
        additional imported inventories.
        """
        
        roadfreight_dataset_names = self.iam_data.roadfreight_markets.coords["variables"].values.tolist()
        railfreight_dataset_names = self.iam_data.railfreight_markets.coords["variables"].values.tolist()
        freight_transport_dataset_names = roadfreight_dataset_names + railfreight_dataset_names
        
        transport_ds = [ds for ds in self.database if ds["name"] in freight_transport_dataset_names]
        
        new_datasets = []
        
        for dataset in list(set([(ds["name"], ds["reference product"]) for ds in transport_ds])):
            new_datasets.extend(
                self.fetch_proxies(
                    subset = transport_ds,
                    name = dataset[0],
                    ref_prod = dataset[1],
                ).values()
            )
        
        for new_ds in new_datasets:
            self.adjust_transport_efficiency(new_ds)
            self.adjust_material_efficiency(new_ds)
            if not self.is_in_index(new_ds):
                self.add_to_index(new_ds)
                self.database.append(new_ds)
                #logger.info(f"New dataset {new_ds['name']} in {new_ds['location']} has been created.")
                
        
    def adjust_transport_efficiency(self, dataset):
        """
        The function updates the efficiencies of transport datasets
        using the transport efficiencies, created in data_collection.py.
        """
        
        vehicles_map = get_vehicles_mapping()
        
        # create a list that contains all energy carrier markets used in transport
        energy_carriers = vehicles_map["energy carriers"]
        
        # create a list that contains all biosphere flows that are related to the direct combustion of fuel
        fuel_combustion_emissions = vehicles_map["fuel combustion emissions"]
        
        # calculate scaling factor 
        if "lorry" in dataset["name"]:
            scaling_factor = 1 / self.find_iam_efficiency_change(
                data=self.iam_data.roadfreight_efficiencies,
                variable=dataset["name"],
                location=dataset["location"],
            )
        elif "train" in dataset["name"]:
            scaling_factor = 1 / self.find_iam_efficiency_change(
                data=self.iam_data.railfreight_efficiencies,
                variable=dataset["name"],
                location=dataset["location"],
            )

        if scaling_factor is None:
            scaling_factor = 1
        
        # rescale exchanges
        if scaling_factor != 1 and scaling_factor > 0:
            rescale_exchanges(
                dataset,
                scaling_factor,
                technosphere_filters=[
                    ws.either(*[ws.contains("name", x) for x in energy_carriers]) # efficiency increase of dataset for train freight transport is also used to update shunting efficiency
                ],
                biosphere_filters=[
                    ws.either(*[ws.contains("name", x) for x in fuel_combustion_emissions])
                ],
                remove_uncertainty=False,
            )
            
            # Update the comments
            text = (
                f"This dataset has been modified by `premise`, according to the energy transport "
                f"efficiencies indicated by the IAM model {self.model.upper()} for the IAM "
                f"region {dataset['location']} in {self.year}, following the scenario {self.scenario}. "
                f"The energy efficiency of the process has been improved by {int((1 - scaling_factor) * 100)}%."
            )
            dataset["comment"] = text + (dataset["comment"] if dataset["comment"] is not None else "")

            if "log parameters" not in dataset:
                dataset["log parameters"] = {}

            dataset["log parameters"].update({"efficiency change": 1 / scaling_factor,})
    
            
    def adjust_material_efficiency(self, dataset):
        """
        This function relinks the exchange of the actual vehicle (for the base year 2020)
        and reconnects it to the vehicle production (with material efficiency) for the year
        the database is created for.
        The code selects inventories for trucks already imported from carculator.
        """
        
        for exc in dataset["exchanges"]:
            if "duty truck" in exc["name"]:
                exc["name"] = exc["name"] + ", " + str(self.year)
        
            
    def generate_transport_markets(self):
        """
        Function that creates market processes and adds them to the database.
        It calculates the share of inputs to each market process and 
        creates the process by multiplying the share with the amount of reference product, 
        assigning it to the respective input.
        Regional market processes then make up the world market processes.
        """
     
        # regional transport markets to be created (keys) with inputs list (values)
        transport_markets_tbc = {
            "market for transport, freight, lorry, unspecified powertrain": 
                self.iam_data.roadfreight_markets.coords["variables"].values.tolist(),
            "market for transport, freight train, unspecified powertrain": 
                self.iam_data.railfreight_markets.coords["variables"].values.tolist(),
        }

        new_transport_markets = []
        
        # create regional market processes
        for markets, vehicles in transport_markets_tbc.items():
            for region in self.iam_data.regions:
                market = {
                    "name": markets, 
                    "reference product": markets.replace("market for ", ""),
                    "unit": "ton kilometer",
                    "location": region,
                    "exchanges": [
                        {
                            "name": markets,
                            "product": markets.replace("market for ", ""),
                            "unit": "ton kilometer",
                            "location": region,
                            "type": "production",
                            "amount": 1,
                        }
                    ],
                    "code": str(uuid.uuid4().hex),
                    "database": "premise",
                    "comment": f"Fleet-average vehicle for the year {self.year}, "
                    f"for the region {region}.",
                }
                
                # add exchanges
                if region != "World":
                    for vehicle in vehicles:
                        if markets == "market for transport, freight, lorry, unspecified powertrain":
                            market_share = self.iam_data.roadfreight_markets.sel(region=region, variables=vehicle, year=self.year).item()
                        elif markets == "market for transport, freight train, unspecified powertrain":
                            market_share = self.iam_data.railfreight_markets.sel(region=region, variables=vehicle, year=self.year).item()
                        
                        # determine the reference product
                        if "lorry" in vehicle:
                            product = "transport, freight, lorry"
                            if "diesel" in vehicle or "compressed gas" in vehicle:
                                euro = ", EURO-VI"
                            else:
                                euro = ""
                        elif "train" in vehicle:
                            product = "transport, freight train"
                            euro = ""

                        if market_share > 0:
                            market["exchanges"].append(
                                {
                                    "name": vehicle,
                                    "product": product + euro,
                                    "unit": "ton kilometer",
                                    "location": region,
                                    "type": "technosphere",
                                    "amount": market_share,
                                }
                                )
                    
                    # add to log
                    self.write_log(dataset=market, status="created")
                    # add it to list of created datasets
                    self.add_to_index(market)
                    
                new_transport_markets.append(market)
        
        
        # world markets to be created
        vehicles_map = get_vehicles_mapping()
        dict_transport_ES_var = vehicles_map["energy service variables"][self.model]["mode"]
        
        dict_regional_shares = {}
        
        # create world market transport datasets exchanges
        for market, var in dict_transport_ES_var.items():
            for region in self.iam_data.regions:
                if region != "World":
                    # calculate regional shares
                    dict_regional_shares[region] = (
                        ( 
                         self.iam_data.data.sel(
                            region=region, 
                            variables=var, 
                            year=self.year).values
                        )/(
                        self.iam_data.data.sel(
                            region="World", 
                            variables=var, 
                            year=self.year).item()
                        )
                    )
        
        # add exchanges    
        for ds in new_transport_markets:
            if ds["location"] == "World":
                for region in self.iam_data.regions:
                    if region != "World":
                        ds["exchanges"].append(
                            {
                                "name": ds["name"],
                                "product": ds["name"].replace("market for ", ""),
                                "unit": "ton kilometer",
                                "location": region,
                                "type": "technosphere",
                                "amount": dict_regional_shares[region],
                            }
                        )
                
        self.database.extend(new_transport_markets)
            
    
    def generate_unspecified_transport_vehicles(self):
        """
        This function generates unspecified transport vehicles for the IAM regions.
        The unspecified datasets refer to a specific size of the vehicle but represent
        an average of powertrain technology for a specific region.
        This only applies to freight lorries so far.
        """
        
        vehicles_map = get_vehicles_mapping()
        dict_transport_ES_var = vehicles_map["energy service variables"][self.model]["size"]
        dict_vehicle_types = vehicles_map["vehicle types"]

        weight_specific_ds = []
        
        # create regional size dependent technology-average markets
        for region in self.iam_data.regions:
            for market, var in dict_transport_ES_var.items():
                if var in self.iam_data.data.variables:
                    vehicle_unspecified = {
                        "name": market, 
                        "reference product": market.replace("market for ", ""),
                        "unit": "ton kilometer",
                        "location": region,
                        "exchanges": [
                            {
                                "name": market,
                                "product": market.replace("market for ", ""),
                                "unit": "ton kilometer",
                                "location": region,
                                "type": "production",
                                "amount": 1,
                            }
                        ],
                        "code": str(uuid.uuid4().hex),
                        "database": "premise",
                        "comment": f"Fleet-average vehicle for the year {self.year}, "
                        f"for the region {region}.",
                    }
                    
                    # add exchanges for regional datasets
                    if region != "World":
                        for vehicle_types, names in dict_vehicle_types.items():
                            variable_key = var + "|" + vehicle_types
                            if variable_key in self.iam_data.data.variables:
                                # calculate regional shares
                                regional_weight_shares = (
                                    ( 
                                    self.iam_data.data.sel(
                                        region=region, 
                                        variables=variable_key, 
                                        year=self.year).values
                                    )/(
                                    self.iam_data.data.sel(
                                        region=region, 
                                        variables=var, 
                                        year=self.year).item()
                                    )
                                )

                                if regional_weight_shares > 0:
                                    if "diesel" in names or "compressed gas" in names:
                                        euro = ", EURO-VI"
                                    else:
                                        euro = ""
                                        
                                    vehicle_unspecified["exchanges"].append(
                                        {
                                            "name": names + ", " + market.split(',')[3].strip() + euro + ", long haul",
                                            "product": "transport, freight, lorry" + euro,
                                            "unit": "ton kilometer",
                                            "location": region,
                                            "type": "technosphere",
                                            "amount": regional_weight_shares,
                                        }
                                    )
                    
                    # add exchanges for global dataset          
                    elif region == "World":
                        for reg in self.iam_data.regions:
                            if reg != "World":
                                    regional_weight_shares = (
                                        ( 
                                        self.iam_data.data.sel(
                                            region=reg, 
                                            variables=var, 
                                            year=self.year).values
                                        )/(
                                        self.iam_data.data.sel(
                                            region="World", 
                                            variables=var, 
                                            year=self.year).item()
                                        )
                                    )
                                    
                                    if regional_weight_shares > 0:
                                        vehicle_unspecified["exchanges"].append(
                                            {
                                                "name": market,
                                                "product": market.replace("market for ", ""),
                                                "unit": "ton kilometer",
                                                "location": reg,
                                                "type": "technosphere",
                                                "amount": regional_weight_shares,
                                            }
                                        )
                                        
                # only add markets that have inputs
                if len(vehicle_unspecified["exchanges"]) > 1:             
                    weight_specific_ds.append(vehicle_unspecified)
                    # logger.info(f"Created dataset SIZE MARKETS {vehicle_unspecified['name']} for region {region}.")

                # add to log
                self.write_log(dataset=vehicle_unspecified, status="created")
                # add it to list of created datasets
                self.add_to_index(vehicle_unspecified)
                
                
        self.database.extend(weight_specific_ds)
                                              
        #TODO: regional unspecified vehicles per driving cycle could have same shares but are not used for markets?


    def empty_ecoinvent_datasets(self):
        """
        The function specifies and deletes inventory datasets.
        In this case transport datasets from ecoinvent, as they
        are replaced by the additional LCI imports.
        """

        vehicles_map = get_vehicles_mapping()
        
        ecoinvent_ds = vehicles_map["ecoinvent freight transport"]
        ds_mapping = vehicles_map["freight transport"][self.model]

        # empty the dataset of all exchanges except the reference product and update comment
        for dataset in self.database:
            if dataset["name"].strip().lower() in (name.strip().lower() for name in ecoinvent_ds):
                dataset["exchanges"] = [exc for exc in dataset["exchanges"] if exc["type"] not in ["technosphere", "biosphere"]]
                
                for key, value in ds_mapping.items():
                    if key in dataset["name"]:
                        dataset["comment"] = f"This dataset has been replaced by the new dataset {value}."
                        break # This can be refined as links are only created to market processes
                    else:
                        dataset["comment"] = f"This dataset has been replaced by a new dataset."

    
    def relink_exchanges(self):
        """
        This function goes through all datasets in the database that use transport, freight (lorry or train) as one of their exchanges.
        It replaced those ecoinvent transport exchanges with the new transport inventories and the newly created transport markets.
        """
        
        vehicles_map = get_vehicles_mapping()
        
        for dataset in self.database:
            if "transport, freight" not in dataset["name"]:
                for exc in ws.technosphere(
                    dataset,
                    ws.contains("name", "transport, freight"),
                    ws.equals("unit", "ton kilometer"),
                    ):
                
                    if any(key.lower() in exc["name"].lower() for key in vehicles_map['freight transport'][self.model]):                 
                        key = [
                            k 
                            for k in vehicles_map['freight transport'][self.model]
                            if k.lower() in exc["name"].lower()
                        ][0]
                        
                        if "input" in exc:
                            del exc["input"]
                            
                        exc["name"] = f"{vehicles_map['freight transport'][self.model][key]}"
                        exc["location"] = self.geo.ecoinvent_to_iam_location(dataset["location"])
                        exc["product"] = (f"{vehicles_map['freight transport'][self.model][key]}").replace("market for ", "")
                        
                        # if exc["name"] == "market for transport, freight, lorry, 40t gross weight, unspecified powertrain":
                        #    # logger.info(f"Replaced exchange in dataset {dataset['name']} with {exc['name']} for region {exc['location']}.")

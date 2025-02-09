"""
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                              <<< SOME NOTES >>>                               #
#                                                                               #
#>>> This script uses the Poisson Generalized Additive Models (GAM) to examine  # 
#    the contribution of the independent variables (aspect, slope, altitude,    #     
#    etc.) on a dependent variable (Malleability) in the forest resilience      #
#    assessment study.                                                          #
#                                                                               #
#>>> Input variables must be in GeoTIFF format.                                 #
#                                                                               #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

@author : Reza Rezaei
email   : rezarezaei2008@gmail.com
version : 1.0
year    : 2024
"""


import os
import numpy as np
import pandas as pd
import numpy.ma as ma
import rasterio as rio
import matplotlib.pyplot as plt
from contextlib import redirect_stdout
from pygam import PoissonGAM, s, l, f, te

import warnings
warnings.simplefilter("ignore", UserWarning)


class PoissionGAM:
    def __init__(self, independent_variables, dependent_variable, area_name, 
                 output_dir):
        self.indep_vars = independent_variables
        self.dep_var = dependent_variable
        self.area = area_name
        self.outdir = output_dir
        self.mask_val = -1000

    def getDataRaster(self, data_name, data_path):
        scaled_data = ["Burn Severity", "Pre-fire NDVI", "NDVI/Burn", list(self.dep_var.keys())[0]] 
        
        with rio.open(self.indep_vars["Altitude"]) as ref_src:
            ref_arr = ref_src.read(1)
            masked_ref_arr = ma.masked_where(ref_arr == self.mask_val, ref_arr)
            with rio.open(data_path) as src:
                data = src.read(1)
                masked_data = np.ma.array(data, mask=masked_ref_arr.mask)
                masked_data = masked_data.flatten()
                masked_data = masked_data.compressed()
                masked_data[masked_data==-1000] = 0
                if data_name in scaled_data:
                    data_scale_factor = float(src.tags(bidx=0)["SCALE_FACTOR"])
                    masked_data = masked_data * data_scale_factor    
        return masked_data 
    
    def Degrees2Direction(self, degrees):
        directions = ['North', 'Northeast', 'East', 'Southeast', 
                      'South', 'Southwest', 'West', 'Northwest']
        degrees = np.asarray(degrees)
        idx = round(degrees / 45) % 8
        return directions[idx]
    
    def MakeDataframe(self):
        data_dict = {}
        for indep_var_name, indep_file in self.indep_vars.items():
            masked_data = self.getDataRaster(data_name=indep_var_name, 
                                             data_path=indep_file)
            data_dict[indep_var_name] = masked_data
            
        for dep_var_name, dep_file in self.dep_var.items():
            masked_data = self.getDataRaster(data_name=dep_var_name, 
                                             data_path=dep_file)
            data_dict[dep_var_name] = masked_data    

        df = pd.DataFrame(data_dict)
        df["Aspect"] = df["Aspect"].apply(self.Degrees2Direction)
        return df
    
    def ModelSummary(self, gam, indep_vars):
        heading = ""
        for count, feature in enumerate(indep_vars):
            txt = f"Feature ({count}) = {feature}\n"
            heading += str(txt)
        heading += f"Dependent variable = {list(self.dep_var.keys())[0]}\n"   
        output_file = f"{self.area}_gam_summary.txt"
        output_path = os.path.join(self.outdir, output_file)
        with open(output_path, "w") as file:
            with redirect_stdout(file):
                print(heading)
                print(gam.summary())
                
    def PartialDependencePlot(self, gam, column_names):
        plt.figure(figsize=(len(column_names) * 11.7, 6))
        for i, term in enumerate(gam.terms):
            if gam.terms[i].isintercept:
                continue 
            plt.subplot(1, len(column_names), i + 1)
            XX = gam.generate_X_grid(term=i)                                    
            plt.plot(XX[:, i], gam.partial_dependence(term=i, X=XX))
            plt.plot(XX[:, i], gam.partial_dependence(term=i, X=XX, 
                                                      width=0.95)[1], 
                     c="r", ls="--")
            plt.xlabel(f"{column_names[i]}", fontsize=38)
            plt.ylabel(list(self.dep_var.keys())[0], fontsize=38)
            
            if f"{column_names[i]}" == "Terrain Aspect":
                categories = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
                plt.xticks(ticks=range(8), labels=categories)  
                
            plt.yticks(fontsize=32)
            plt.xticks(fontsize=32)
            
            ax = plt.gca()
            for spine in ax.spines.values():
                spine.set_color("black")
                spine.set_linewidth(2)
            
            xlim = plt.gca().get_xlim()   
            ylim = plt.gca().get_ylim()
            x_padding = 0.17 
            y_padding = 0.05
            x = xlim[0] + x_padding * (xlim[1] - xlim[0])
            y = ylim[1] - y_padding * (ylim[1] - ylim[0])
            plt.text(x, y, f"{self.area}", fontsize=36, 
                     ha="right", va="top", weight="bold",
                     bbox = dict(facecolor = "red", alpha = 0.5))
            plt.tight_layout(pad=1)
        plt.subplots_adjust(hspace=1) 
        
        fig_name = f"{self.area}_partial_dependence_plots.png"
        output_path = os.path.join(self.outdir, fig_name)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
    
    def DrawDevienceResidualsPlot(self, prediction, devience_residuals):
        plt.figure(figsize=(8, 6))
        plt.scatter(prediction, devience_residuals, alpha=0.6)
        plt.axhline(0, linestyle="--", color="red")
        plt.xlabel("Predicted Values", fontsize=14)
        plt.ylabel("Deviance Residuals", fontsize=14)
        title = "Deviance Residuals Plot"
        plt.title(title, fontsize=16)
        fig_name = f"{self.area}_" + title.replace(" ", "_") + ".png"
        output_path = os.path.join(self.outdir, fig_name)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
    
    def PoissionGAM(self):
        df = self.MakeDataframe()
        df.loc[df[list(self.dep_var.keys())[0]] < 0, list(self.dep_var.keys())[0]] = np.nan
        df.loc[df["Burn Severity"] < 0, "Burn Severity"] = np.nan
        df = df.dropna()  
        
        custom_mapping = {"North": 0, "Northeast": 1, "East": 2, "Southeast": 3,
                          "South": 4, "Southwest": 5, "West": 6, "Northwest": 7}
        df["Terrain Aspect"] = df["Aspect"].map(custom_mapping)

        X_df = df[["Terrain Aspect", "Slope", "Altitude", "Pre-fire NDVI", "Burn Severity"]]  
        y_df = df[list(self.dep_var.keys())[0]]
        X = X_df.to_numpy()
        y = y_df.to_numpy()
        
        gam = PoissonGAM(f(0) + s(1) + s(2) + s(3) + s(4)).fit(X, y) 

        self.ModelSummary(gam=gam, indep_vars=list(X_df.columns))        
        self.PartialDependencePlot(gam=gam, column_names=list(X_df.columns))
        self.DrawDevienceResidualsPlot(prediction=gam.predict(X), 
                                       devience_residuals=gam.deviance_residuals(X, y))
        
        


#===================================== RUN ====================================
area_name = "Adana_01"
indep_vars = {"Aspect": "E:/My_articles/6- Wildfire/Data/DEM/Slope_aspect/Adana/AD01_aspectRaster.tif",
              "Slope": "E:/My_articles/6- Wildfire/Data/DEM/Slope_aspect/Adana/AD01_slopeRaster.tif",
              "Altitude": "E:/My_articles/6- Wildfire/Data/DEM/Slope_aspect/Adana/resampled_AD01_masked_DEM.tif",
              "Pre-fire NDVI": "E:/My_articles/6- Wildfire/Data/Indicators_revised/Indicators/AD/NDVI/cropped_AD01_Pre-fire_NDVI_20210724T081609_20m.tif",
              "Burn Severity": "E:/My_articles/6- Wildfire/Data/Indicators_revised/Indicators/AD/NBR/AD01_dNBR_20210823.tif"}
dep_var = {"Malleability": "E:/My_articles/6- Wildfire/Data/Indicators_revised/Indicators/AD/NDVI/AD01_malleability_20240728.tif"}
                            
out_path = "E:/My_articles/6- Wildfire/Data/GAM_outputs/AD"


ins = PoissionGAM(independent_variables=indep_vars,
                  dependent_variable=dep_var,
                  area_name=area_name,
                  output_dir=out_path)

ins.PoissionGAM()
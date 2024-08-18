"""
pykino gam(mgcv) Class
=================
Main class to wrap R's mgcv library
"""
from copy import copy
from rpy2.robjects.packages import importr
import rpy2.robjects as robjects
from rpy2.rinterface_lib import callbacks
import rpy2.rinterface as rinterface
from rpy2.robjects.conversion import localconverter
from rpy2.robjects import numpy2ri
from rpy2.robjects import pandas2ri
import warnings
import traceback
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from rpy2.robjects.vectors import StrVector

from ..concerto import con2R, pandas2R, R2pandas, R2numpy
from pandas.api.types import CategoricalDtype
import matplotlib.pyplot as plt
from IPython.display import Image, display

# Import R libraries we need
base = importr("base")
stats = importr("stats")
# Make a reference to the default R console writer from rpy2
consolewrite_warning_backup = callbacks.consolewrite_warnerror
consolewrite_print_backup = callbacks.consolewrite_print

# Activate Pandas and R dataframe conversion
pandas2ri.activate()

class gam(object):
    def __init__(self, formula, data, family = "gaussian()"):
        self.formula = formula
        self.data = data
        self.plot_filename = 'model_plot.png'  # Initialize plot filename
        
        #self.family=family
        
        self.mgcv = importr('mgcv')
        self.stats = importr('stats')
        self.family=family
        # Determine if the family is from mgcv or stats
        if family in ['scat', 'cox.ph', 'gaulss', 'gevlss', 'gumblss', 'mvn', 'multinom', 'nb', 'negbin', 'ocat', 'shash']:
            self.family_func = getattr(self.mgcv, family)()
        else:
            try:
                self.family_func = getattr(self.stats, family)()
            except AttributeError:
                raise ValueError(f"Family '{family}' is not recognized. Please check the family name.")

        # Optionally print the family to confirm
        print(f"Loaded family: {family}")


    def gam(self, weight=None, discrete=None, method=None):
        with localconverter(robjects.default_converter + pandas2ri.converter):
            r_data = robjects.conversion.py2rpy(self.data)
        
        args = {'data': r_data}
        
        if weight is not None:
            weight_series = pd.Series(weight)  
            with localconverter(robjects.default_converter + pandas2ri.converter):
                r_weight = robjects.conversion.py2rpy(weight_series)
            args['weights'] = r_weight
        args['family'] = self.family
        if discrete is not None:
            args['discrete'] = robjects.BoolVector([discrete])
        if method is not None:
            args['method'] = robjects.StrVector([method])
        
        model = self.mgcv.gam(robjects.Formula(self.formula), **args)
        
        return model
    

    def bam(self, weight=None, discrete=None, method=None):
        with localconverter(robjects.default_converter + pandas2ri.converter):
            r_data = robjects.conversion.py2rpy(self.data)
        
        args = {'data': r_data}
        
        if weight is not None:
            weight_series = pd.Series(weight)  
            with localconverter(robjects.default_converter + pandas2ri.converter):
                r_weight = robjects.conversion.py2rpy(weight_series)
            args['weights'] = r_weight
        args['family'] = self.family
        if discrete is not None:
            args['discrete'] = robjects.BoolVector([discrete])
        if method is not None:
            args['method'] = robjects.StrVector([method])
        
        model = self.mgcv.bam(robjects.Formula(self.formula), **args)
        
        return model  

    '''
    #def predict(self, model, newdata):
        # Predict using model
        with localconverter(robjects.default_converter + pandas2ri.converter):
            r_newdata = robjects.conversion.py2rpy(newdata)
        
        predictions = robjects.r['predict'](model, newdata=r_newdata)
        
        #Convert predictions to pandas DataFrame
        with localconverter(robjects.default_converter + pandas2ri.converter):
            predictions_df = robjects.conversion.rpy2py(predictions)
        
        return predictions_df
    '''



    def summary(self, model):
        # Get summary of the GAM/BAM model
        summary = robjects.r['summary'](model)

        return summary

    
    def gamv(self, formula, knots, data, family, nsim):
        # Convert pandas DataFrame to R data frame
        with localconverter(robjects.default_converter + pandas2ri.converter):
            r_data = robjects.conversion.py2rpy(data)
        
        # Prepare additional arguments
        if knots:
            r_knots = robjects.ListVector({k: robjects.FloatVector(v) for k, v in knots.items()})
        else:
            r_knots = robjects.NULL
        
        # Fit GAMM using gam() function in mgcv
        formula_r = robjects.Formula(formula)
        model = self.mgcv.gam(formula_r, knots=knots , data=r_data, family=family, aGam=robjects.ListVector({'knots': r_knots}), aViz=robjects.ListVector({'nsim': nsim}))

        return model
    
    '''
        # Convert DataFrame to R data.frame
        with localconverter(robjects.default_converter + pandas2ri.converter):
            r_newdata = robjects.conversion.py2rpy(newdata)
        
        # Make predictions
        if se_fit:
            predictions = robjects.r['predict'](model, newdata=r_newdata, type=type, se_fit=True)
            # Extract predictions and standard errors
            with localconverter(robjects.default_converter + pandas2ri.converter):
                pred_df = pd.DataFrame({
                    'fit': robjects.conversion.rpy2py(predictions[0]),
                    'se.fit': robjects.conversion.rpy2py(predictions[1])              
            })
        else:
            predictions = robjects.r['predict'](model, newdata=r_newdata, type=type)
            with localconverter(robjects.default_converter + pandas2ri.converter):
                pred_df = pd.DataFrame({
                    'fit': robjects.conversion.rpy2py(predictions)
            })
        
        return pred_df
    '''
    def predict(self, model, newdata, type='response', se_fit=False):
        """
        Predict using the fitted model.
        
        Parameters:
        - newdata: pandas DataFrame containing the new data for prediction.
        - type: Type of prediction. Options are 'link', 'response', 'terms'.
        - se_fit: Boolean indicating whether to compute standard errors.
        
        Returns:
        - A pandas DataFrame containing the predictions (and optionally standard errors).
        """
        # Convert DataFrame to R data.frame
        with localconverter(robjects.default_converter + pandas2ri.converter):
            r_newdata = robjects.conversion.py2rpy(newdata)
    
        if se_fit:
            # Perform prediction with standard errors
            predictions = robjects.r['predict.gam'](model, newdata=r_newdata, type=type, se_fit=True)
            #print(len(predictions))
            # Extract predictions and standard errors
            fit_values = robjects.conversion.rpy2py(predictions[0])  # First element: fit
            se_fit_values = robjects.conversion.rpy2py(predictions[1])  # Second element: se.fit
            
            #print(se_fit_values)  # Should be list or ndarray
            
            # Create DataFrame with fit and se.fit
            pred_df = pd.DataFrame({
                'fit': fit_values,
                'se.fit': se_fit_values              
            })
        else:
            # Perform prediction without standard errors
            predictions = robjects.r['predict'](model, newdata=r_newdata, type=type)
            fit_values = robjects.conversion.rpy2py(predictions)
            
            # Convert to list if it is a numpy array
            if isinstance(fit_values, np.ndarray) and fit_values.ndim == 1:
                fit_values = fit_values.tolist()
                
            # Create DataFrame with only fit
            pred_df = pd.DataFrame({
                'fit': fit_values
            })
        
        return pred_df
    '''
    def plot(self, model, all_terms=True, pages=1, se=False):

        # Use R's plot.gam function
        plot_func = robjects.globalenv['plot.gam']
        plot_func(self, model, all_terms=all_terms, pages=pages, se=se)
    '''

   
def plot(self, model, all_terms):

        #robjects.globalenv['model'] = model
        robjects.r(f"png('{self.plot_filename}')")

        with localconverter(robjects.default_converter + pandas2ri.converter):
            all_terms_r = robjects.BoolVector([all_terms])
        plot_args = {'all.terms': all_terms_r}
        
        robjects.r['plot.gam'](model, **plot_args)
        
        robjects.r['x11()']

        display(Image(filename=self.plot_filename))

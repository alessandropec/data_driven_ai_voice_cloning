import pandas as pd
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
import typing
from hello.dataWorld.dataReduction import reduce_data
import typing
import seaborn as sns

def set_fig_size(figsize_inches):
    plt.gcf().set_size_inches(figsize_inches[0], figsize_inches[1])

def plt_targets_distr(df: pd.DataFrame,cols: typing.List[str],
                      kind="bar",
                      alpha=0.75,rot=45,
                      fontsize=18,figsize=(20,30),
                      suptitle="Targets distribution for each category") -> typing.Dict[str,pd.DataFrame]:
 
    """
    Plot the distribution of the targets columns.

        Parameters:
        ----------
        df: DataFrame
            Table of data.
        
        cols: list[str]
            The name of targets columns.

        kind: string 
            See matplotlib for full documentation ("bar","line","kde", etc...).

        alpha: float [0,1] 
            Opacity of {bar,line,kde,etc...}.

        rot: int [0,360] 
            Degree rotation of xtick labels.

        fontsize: int 
            Dimension of text.

        figsize: tuple 
            (width,height) of figure.

        suptitle: string 
            Title of figure.
        
        Return:
        ----------
        A dict[str,DataFrame] containing N (number of cols) DataFrames of the distribution, use the name of columns as index.
  
    """
    num_col=len(cols)
    fig,axs=plt.subplots(num_col)

    fig.suptitle(suptitle,fontsize=fontsize)

    dfs={}
    if num_col>1:
        for i,col in enumerate(cols):
            dfs[col]=df.groupby(col)[col].count()
            dfs[col].plot(ax=axs[i],kind=kind,alpha=alpha,rot=rot,fontsize=fontsize,figsize=figsize)
    elif num_col==1:
        dfs[cols[0]]=df.groupby(cols[0])[cols[0]].count()
        dfs[cols[0]].plot(ax=axs,kind=kind,alpha=alpha,rot=rot,fontsize=fontsize,figsize=figsize)
    
    plt.show()
    
    return axs,dfs


def plt_redu_data(df:pd.DataFrame,reduce_col:list=None,
                  algorithm="PCA",y:pd.DataFrame=None,
                  hue:str=None,size:str=None,style:str=None,alpha:float=1,
                  random_state=42,
                  n_components:int=2):
    """
    Given a numerical dataframe with more then 2/3 dimension, reduce it and scatter.


    Parameters:
    -------------
    df: DataFrame
        The data table to reduce.
    
    reduce_col: list
        The column names to reduce
    
    hue: list or str
        Specify key in df to produce different color based on the categories.

    size: list or str
        Specify key in df to produce different seize based on the categories.

    style: list or str
        Specify key in df to produce different style based on the categories.
    
    algorithm: string
        The algorithm used to reduction: ["PCA","LDA","TruncatedSVD"].
        LDA is used for classification, see y parameters.
    
    y: DataFrame
        Only for LDA algorithm, the ground truth of the data.

  
    random_state: int
        Seed to the random generator.  
    
    reduce: boolean
        If False the method expect an already reduced dataset, the reduction process will be skipped. 
    
    n_components: int
        The number of output dimension 1D, 2D (3D work in progress...)

    Return:
    --------------
        A tuple containing the Axes of plot and the reduced DataFrame   
    
    """
  
    #df=df.copy()
    if n_components>2:
        print("Error: a valid value for n_components is 1 or 2, 3 to do...")
        return 

    if reduce_col!=None:
        redu_df=reduce_data(df[reduce_col],algorithm,y,n_components,random_state)[0]
        df[algorithm+"0"]=redu_df[0]
        if n_components>1:
            df[algorithm+"1"]=redu_df[1]
 
  
    if n_components>1:
        ax=sns.scatterplot(data=df,x=algorithm+"0",y=algorithm+"1",hue=hue,size=size,style=style,alpha=alpha)
    else:
        ax=sns.stripplot(data=df,x=algorithm+"0",y=hue,hue=hue,alpha=alpha)

    plt.show()
    return ax,df
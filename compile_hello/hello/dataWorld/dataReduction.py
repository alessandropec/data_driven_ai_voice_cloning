from sklearn.decomposition import PCA,TruncatedSVD
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import pandas as pd
from typing import Tuple


def reduce_data(df:pd.DataFrame,algorithm="PCA",y:pd.DataFrame=None,n_components=2,random_state=42) -> Tuple[pd.DataFrame,PCA]:
    """
    Given a numerical dataframe reduce the data to a lower dimension.

    Parameters:
    -------------
    df: DataFrame
        The data table to reduce.
    
    algorithm: string
        The algorithm used to reduction: ["PCA","LDA","TruncatedSVD"].
        LDA is used for classification, see y parameters.
    
    y: DataFrame
        Only for LDA algorithm, the ground truth of the data.

    n_components: int
        The number of output dimension.
        For PCA algorithm <= n_classes
        For LDA algorithm, number of components < n_classes - 1.

    random_state: int
        Seed to the random generator.  

    Return:
    --------------
        A dataframe containing the data reduced to n_component.  
    
    """
    if algorithm=="PCA":
        pca=PCA(n_components=n_components,random_state=random_state)
        out=pca.fit_transform(df)
        return pd.DataFrame(out),pca

    if algorithm=="LDA":
        lda=LDA(n_components=n_components)
        out=lda.fit_transform(df,y)
        return pd.DataFrame(out),lda

    if algorithm=="TruncatedSVD":
        tsvd=TruncatedSVD(n_components=n_components)
        out=tsvd.fit_transform(df)
        return pd.DataFrame(out),tsvd
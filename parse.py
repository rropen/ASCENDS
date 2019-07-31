import argparse
from pathlib import PurePath 
import pandas as pd
import csv

def main(args):


    """
    Load data
    """
    try:
        train_type=args.train_type
        project_file=PurePath(args.project_file)
        csv = pd.read_csv(project_file / "metadata.csv",index_col=None)
        csv.drop(['Unnamed: 0'], axis=1)
    except Exception as e:
        print('err loading data')
        print(e)

    """
    Parse data
    """
    r2=[]
    mae=[]
    acc=[]
    for i in csv['model_stats']:
        r2.append(eval(i)['R2'])
        mae.append(eval(i)['MAE'])
        acc.append(eval(i)['Accuracy'])
    csv['R2']=r2
    csv['MAE']=mae
    csv['Accuracy']=acc
    arglist=[]
    for i in csv['model_args']:
        if not i in arglist:
            arglist.append(i)
    for i in arglist:
        matches=csv.loc[csv['model_args']==i]
        print("\nArguments: %s" % i)
        print("R2 mean: %s" % matches['R2'].mean())
        print("R2 std dev: %s" % matches['R2'].std())
    argdata=pd.DataFrame(columns=['Arguments','R2 Mean','R2 Std Dev','MAE Mean','MAE Std Dev','Accuracy Mean','Accuracy Std Dev'])
    for i in arglist:
        matches=csv.loc[csv['model_args']==i]
        if train_type=='r':
            argdata=argdata.append(pd.DataFrame([[i,matches['R2'].mean(),matches['R2'].std(),matches['MAE'].mean(),matches['MAE'].std(),None,None]],columns=['Arguments','R2 Mean','R2 Std Dev','MAE Mean','MAE Std Dev','Accuracy Mean','Accuracy Std Dev']),ignore_index=True,)
        else:
            argdata=argdata.append(pd.DataFrame([[i,None,None,None,None,matches['Accuracy'].mean(),matches['Accuracy']].std()],columns=['Arguments','R2 Mean','R2 Std Dev','MAE Mean','MAE Std Dev','Accuracy Mean','Accuracy Std Dev']),ignore_index=True)
    argdata.to_csv(project_file / "parseddata.csv" )
    

if __name__=="__main__":
    
    print("\n * ASCENDS: Advanced data SCiENce toolkit for Non-Data Scientists ")
    print(" * Metadata parser \n")
    print(" programmed by Matt Sangkeun Lee (lees4@ornl.gov) ")

    parser = argparse.ArgumentParser()
    parser.add_argument("train_type", help="Choose training type: 'c' for classification or 'r' for regression.",choices=['c','r'])
    parser.add_argument("project_file", help="project file to parse")

    args = parser.parse_args()
    main(args)



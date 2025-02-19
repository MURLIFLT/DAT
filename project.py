import pandas as pd
import pandas_profiling
from sklearn import svm
from sklearn.calibration import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
import streamlit as st
from streamlit_pandas_profiling import st_profile_report
from pandas_profiling import ProfileReport
import numpy as np
import ydata_profiling
import plotly.express as px
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso, LinearRegression, LogisticRegression, ridge_regression
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn import datasets, linear_model, metrics
import seaborn as sns
import matplotlib.pyplot as plt
import ydata_profiling
from sklearn.preprocessing import LabelEncoder


st.set_page_config(layout="wide")

def stateful_button(*args, key=None, **kwargs):
    if key is None:
        raise ValueError("Must pass key")

    if key not in st.session_state:
        st.session_state[key] = False

    if st.button(*args, **kwargs):
        st.session_state[key] = not st.session_state[key]

    return st.session_state[key]

with st.sidebar:
    st.header("**Import Data**")
    fileMethod = st.radio("**Select any one method to fetch data:**",options=["Browse Files","Link/Name"])

    if fileMethod=="Browse Files":
        dataupload = st.file_uploader("**File Upload:**",key="dfg")
        if dataupload is not None:
            df = pd.read_csv(dataupload)
            st.dataframe(df,width=300)
    
    try:    
        if fileMethod=="Link/Name":
            loadingData = st.text_input("**Enter Name of dataset to fetch the data from seaborne:** ",key="fetch")
            if loadingData is not None:
                df = sns.load_dataset(loadingData)
                st.dataframe(df,width=300) 
    except :
        ValueError()

    if stateful_button("***Follow Me***👍😎",key="foll"):
        linkedin = st.image("linkedin.png")
        # st.write("[Linkedin](https://www.linkedin.com/in/sandipsingh-/)")
        insta = st.image("insta.png")
        # st.write("[Instagram](https://www.instagram.com/sandip888singh/)")
        youtube = st.image("youtube.png")
        # st.write("[Youtube](https://www.youtube.com/@programmingbird2896)")
    
try:
    with st.expander("**Uploaded/Original Data**"):
        st.dataframe(df)
except (NameError):
    ()
st.title("Data Analysis and Prediction Tool")     


def checking(nullCheck):
    check = df[nullCheck].isnull().sum() 
    if check == 0:
        st.warning("No Null Values to write")    
    return

marr = []
varr = []
   
def missingVal(m):
    mcounter = 100
    vcounter = 500
    coli = 1
    for j in range(0,m):
        missingColName = st.selectbox(f"**:red[Select column {coli}:]** ",df.columns,key=mcounter)
        nullCheck = missingColName
        checking(nullCheck)
        marr.append(missingColName)
        missingValues = st.text_input(f"**:blue[Enter value for column {coli}:]** ",key=vcounter)
        varr.append(missingValues)
        coli+=1
        mcounter+=1
        vcounter+=1
    return    

res = {}
def createDict():
    for key in  marr:
        for value in varr:
            res[key] = value
            varr.remove(value)
            break
    return       

rVal = []
def replaceVal(p):
    temper = 145
    for r in range(0,p):
        rCol = st.selectbox(f"Select column {r}:",df.columns,key=temper)
        nullCheck = rCol
        checking(nullCheck)
        rVal.append(rCol)
        temper+=1
    return

arr = []
def uniques(n):
        counter = 1
        for i in range(0,n):
            forUnique = st.selectbox(f"Enter the colName{i} : ",options=df.columns,key=counter+45)
            arr.append(forUnique)
            counter+=1
        return

def graphs():
       
    xAxis = st.selectbox("**Select X axis value:** ",options=(df.columns))
    yAxis = st.selectbox("**Select Y axis value:** ",options=(df.columns))
    # if xAxis=="None":
    #     st.write(type("None"))
    pltMethod = st.radio("**Select the plot type:👇**",options=("Line","Pair Plot","Scatter","Bar","Funnel","Pie","Histogram","Area","Box","Violin","Strip","ECDF","Density Heatmap","Density Contour","3D Scatter","3D Line","Scatter Matrix"),horizontal=True)
    
    title = st.text_input("**Enter title of graph:** ")
    select = st.radio("**want color column?**",("Yes","No"))
    if select=="Yes":
        color = st.selectbox("**Select the column name for color:** ",options=(df.columns),key="color")
    if select=="No":
        color = None  
    
    try:
        if pltMethod=="Line":
            plot =  px.line(df,x=xAxis,y=yAxis,title=title,color=color,markers=True)
        if pltMethod=="Pair Plot":
            plotsns =  sns.pairplot(df)
            st.pyplot(plotsns)
        if pltMethod=="Scatter":
            plot =  px.scatter(df,x=xAxis,y=yAxis,title=title,color=color)
        if pltMethod=="Bar":
            plot =  px.bar(df,x=xAxis,y=yAxis,title=title,color=color)
        if pltMethod=="Funnel":
            plot =  px.funnel(df,x=xAxis,y=yAxis,title=title,color=color)
        if pltMethod=="Pie":
            plot =  px.pie(df,values=xAxis,names=yAxis,title=title,color=color)
            st.info("Select as: Xaxis=value & Yaxis=name")
        if pltMethod=="Histogram":
            plot =  px.histogram(df,x=xAxis,y=yAxis,title=title,color=color)
        if pltMethod=="Box":
            plot =  px.box(df,x=xAxis,y=yAxis,title=title,color=color)  
        if pltMethod=="Area":
            plot =  px.area(df,x=xAxis,y=yAxis,title=title,color=color) 
        if pltMethod=="Violin":
            plot =  px.violin(df,x=xAxis,y=yAxis,title=title,color=color,box=True)                    
        if pltMethod=="Strip":
            plot =  px.strip(df,x=xAxis,y=yAxis,title=title,color=color) 
        if pltMethod=="ECDF":
            plot =  px.ecdf(df,x=xAxis,y=yAxis,title=title,color=color)
            st.info("Any one axis must have numerical values.")
        if pltMethod=="Density Heatmap":
            plot =  px.density_heatmap(df,x=xAxis,y=yAxis,title=title,marginal_x="histogram", marginal_y="histogram",text_auto=True)
        if pltMethod=="Density Contour":
            plot =  px.density_contour(df,x=xAxis,y=yAxis,title=title,color=color)
        if pltMethod=="3D Scatter":
            zAxis = st.selectbox("**Select Z axis value**:",options=(df.columns),key="3ds")
            plot =  px.scatter_3d(df,x=xAxis,y=yAxis,z=zAxis,title=title,color=color)
        if pltMethod=="3D Line":
            zAxis = st.selectbox("**Select Z axis value**:",options=(df.columns),key="3dl")
            plot =  px.line_3d(df,x=xAxis,y=yAxis,z=zAxis,title=title,color=color)    
        if pltMethod=="Scatter Matrix":
            plot =  px.scatter_matrix(df)
        st.plotly_chart(plot)
    except (UnboundLocalError):
        ()
    return 

def encoding(ecols): 
    ecounter = 1000
    ecoli = 1
    for e in range(0,ecols):
        eSelecter = st.selectbox(f"Select {ecoli}",df.columns,key=ecounter)
        st.write(eSelecter)
        uniqueForReplace = df[eSelecter].unique()
        vCount = len(uniqueForReplace)
        st.write("Unique:",uniqueForReplace)
        st.write("Total Unique values:",vCount)
        err = []
        for enc in range(0,vCount):
            err.append(enc)
        df[eSelecter] = df[eSelecter].replace(uniqueForReplace,err)
        ecoli+=1
        ecounter+=1 
    return

cvListIndependent = []
cvListDependent = []
modelList = []

def models():
    global modelType
    selectModelType = st.radio("**Select any one Regression or Classification method:**",options=["Random Forest Classifier","Naive Bayes Classifier","Decision Tree Classifier","K Nearest Neighbors Classifier","Support Vector Machine","Linear Regression","Logistic Regression"])
    if selectModelType=="Random Forest Classifier":
        modelType = RandomForestClassifier()
    if selectModelType=="Naive Bayes Classifier":
        modelType = GaussianNB()
    if selectModelType=="Decision Tree Classifier":
        modelType = DecisionTreeClassifier()
    if selectModelType=="K Nearest Neighbors Classifier":
        modelType = KNeighborsClassifier()  
    if selectModelType=="Support Vector Machine":
        modelType = svm.SVC()
    if selectModelType=="Linear Regression":
        modelType = LinearRegression()
    if selectModelType=="Logistic Regression":
        modelType = LogisticRegression()  
            
    st.write(f"**You selected : :blue[{selectModelType}]**")

    modelXInput = st.multiselect("**Select Independent Values (X):**",df.columns,key="xpred")
    modelYInput = st.multiselect("**Select Dependent Values (Y):**",df.columns,key="ypred")
    st.write(modelXInput)
    st.write(modelYInput)
    for i in modelXInput:
        cvListIndependent.append(i)
    for j in modelYInput:
        cvListDependent.append(j)
    X = df[modelXInput]
    st.write("Independent columns (X):",X)
    y = df[modelYInput] 
    st.write("Dependent columns (Y):",y)
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=30)
 
    cols1,cols2 = st.columns(2)  
    with cols1:
        st.write("Training Dataset:")
        st.write("X_train:",X_train)
        st.write("y_train:",y_train)
    with cols2:
        st.write("Testing Dataset:")
        st.write("X_test:",X_test)
        st.write("y_test:",y_test)
    
    model = modelType
    modelList.append(modelType)
    st.write("Model Type:")
    model.fit(X_train, y_train)
    st.write(model)
    
    y_pred = model.predict(X_test)
    
    # st.text("Pred")
    st.write("Y prediction: ",y_pred)
    
    accuracy = accuracy_score(y_test, y_pred)
    st.write(f"**Accuracy: :green[{accuracy}]**","or", f"**:green[{accuracy*100}]%**")
    
    st.write("**Report:**")
    st.dataframe(classification_report(y_test,y_pred, output_dict=True))
    
    # X_new =[[3, 2, 1, 0.2]]
    # prediction = model.predict(X_new)
    # st.write("Prediction of Species: {}".format(prediction))
    
def crossValidation():
    col1,col2,col3 = st.columns(3)
    with col1:
        userData = []
        st.write("**Inputs:**")
        independent = st.dataframe(cvListIndependent)
        st.write("**Prediction:**")
        dependent = st.dataframe(cvListDependent)
    
    with col2:
        i = len(cvListIndependent)
        cvKey = 2
        for index in  range(0,i):
            labels = st.write(f"**{cvListIndependent[index]}:**")
            # for j in range(0,i):
            out = float(st.text_input("Enter the value: ",key=cvKey+4))
            userData.append(out)
            st.write(f"✔ **{out}**")
            cvKey=cvKey+4  
    
    with col3:
        st.write("**Selected Inputs:**")
        st.dataframe([userData])
        # res = [eval(k) for k in userData]
        # st.write(res)
        st.write(modelType)
        pred = modelType.predict([userData])
        st.write(f"**Prediction:** :red[{pred}]")
        st.success("Predicted! ✅")
        

     
tab1,tab2,tab3,tab4,tab5 = st.tabs(["**Data Overview**","**Data Cleaning**","**Data Visualization**","**Model Building**","**Model Testing**"])

try:
    
    with tab1:
        if stateful_button("**Data Profile**",use_container_width=True,key="pp"):
            # st.write(pp.ProfileReport(df))
            st.info("Close the data profile after use, for better experience")
            profile = ProfileReport(df,title="Data info")
            x=st_profile_report(profile)
            st.write(x)
            
    with tab2:
        st.header("Data Cleaning 🧹")
        
        if stateful_button("**Missing Values**",use_container_width=True,key="nill"):
            t2col1,t2col2 = st.columns(2)
            with t2col1:
                totalNull = df.isnull().sum()
                st.write("**Total Null Values:**",totalNull)
            with t2col2:
                try:
                    columnName =st.selectbox("**Filter Null values by column Name:**",(df.columns))
                    indexNill = np.where(df[columnName].isnull())[0]
                    st.write(indexNill)
                except KeyError:
                    ValueError()
                
            if stateful_button("**Drop Columns**",use_container_width=True,key="cmnDrop"):
                ma = st.multiselect("**Select the columns to Drop:**",df.columns,key="ColDrop")
                st.write(ma)
                df = df.drop(ma,axis=1)
                st.write(df)
            
            option=st.selectbox("**Select any one method to handle Null values 👇:**",("...","Drop the data","Input missing data","Replace the values"))
            st.write(f"**You selected** :blue[**{option}**] ")
            if option=="Drop the data":
                df = df.dropna()
                st.success(f"**Dropped rows containing null values from column [ {totalNull} ]:**")
                st.write(df)
                st.write(df.index)
                st.write(f"**New records: :green[{len(df)}]** ")
            if option=="Input missing data":
                try:
                    m = st.slider("**Select the total number of columns to enter the missing value:** ",min_value= 1,max_value= 25,step=1,key="inputNew")
                    missingVal(m)   
                    createDict()
                    df = df.fillna(res)
                    st.write(df)
                except (KeyError ,NameError,ValueError):
                    st.error("**Enter column name!**")   
            if option=="Replace the values":
                st.info("Mean & Median must have numerical values!")  
                
                methods = st.radio("**Select Any one Method:**",("Mean","Median","Mode"))

                try:
                    if methods=="Mean":
                        p = st.slider("**Select the columns**: ",0,5,1,key="vf")
                        replaceVal(p)
                        # check = df[rVal].isnull().sum() 
                        # for l in check:
                        #     ad = l
                        # if l == 0:
                        #     st.warning("**No Null values to write / Select numerical columns only**")
                        # try:
                        meanVal=df[rVal].mean()
                        st.write("Mean: ",meanVal)
                        df[rVal]=df[rVal].replace(np.NaN,meanVal)
                        st.write(df)
   
                    elif methods=="Median":
                        p = st.slider("**Select the columns**: ",0,5,1,key="vf")
                        replaceVal(p)
                        medianVal = df[rVal].median()
                        st.write("Median: ",medianVal)
                        df[rVal] = df[rVal].replace(np.NaN,medianVal)
                        st.write(df)
                    
                    elif methods=="Mode":
                        ml = st.slider("**Select the columns**: ",0,25,1)
                        mcounter = 200
                        for modeloop in range(0,ml):
                            modeStore = st.selectbox("Select cols:",df.columns,key=mcounter)
                            modeVal  = df[modeStore].mode()[0]
                            st.write("Mode: ",modeVal)
                            df[modeStore] = df[modeStore].fillna(modeVal)
                            mcounter+=1
                        st.write(df)    
                
                except ValueError:
                    st.error("Please Select different column")               
                except TypeError:
                    st.warning("Please Enter only Numerical Value")        
                # except (KeyError ,NameError):
                #     st.error("Enter column name!")
                    
            if stateful_button("**Encoding Values**",use_container_width=True,key="code"):
                try:
                    ecols = st.slider("**Select the number of columns to Encode:**",1,25,1)
                    encoding(ecols)
                    st.write(df)
                except:
                    TypeError()
            
            
        try:          
            if stateful_button("**Duplicate**",use_container_width=True,key="copy"):
                totalDuplicate = df.duplicated().sum()
                st.write("Total Duplicate Rows: ",  totalDuplicate)
                a = df[df.duplicated()]
                st.write("Duplicate Rows: ",a)
                
                st.subheader("Filter duplicates column  wise")
                forSubset = st.selectbox("Select column name:",df.columns)
                df = (df[df.duplicated(subset=forSubset)])
                st.write(df.index)
                st.write("Total duplicates in column "+forSubset+" is :",len(df))
                st.write(df)
            if stateful_button("**Drop Duplicates**",use_container_width=True,key="drap"):
                dropped = df.drop_duplicates(inplace=True)
                st.success("Duplicate rows dropped  successfully")

        except (KeyError,TypeError):
            ValueError()  
        
        
        dl = df.to_csv()
        st.download_button(label="**Download data as csv**",data=dl,file_name='datacv.csv',mime='text/csv',use_container_width=True,key="downloading")
        
        try:    
            if stateful_button("**Unique**",use_container_width=True,key="colsu"):
                n = st.slider("How many number of columns to find the unique values?",0,50,1)
                uniques(n)
                newdf = df[arr].nunique()
                st.write(newdf)
        except KeyError:
            ValueError()    

        
    with tab3: 
        st.header("Visualization📈📊📉")
        st.subheader("Dataset: ")
        st.write(df)       
        try:
            st.write("Columns in Dataset are:",df.columns) 
            graphs()
        except ValueError:
            ValueError()    
    
    with tab4:
        try:
            st.header("Model Building 👨‍💻🧩")
            models()
        except:
            ValueError()
            
    with tab5:
        try:
            st.header("Model Testing 🕵️")
            crossValidation()
        except:
            ValueError()
    
except (KeyError ,NameError):
        st.error("Please upload dataset!")

    
            
   

    


import streamlit as st
import joblib
import pandas as pd
import sklearn
import imblearn

#load model and features names
Model= joblib.load("Loans_Model_Final.pkl")
Inputs= joblib.load("Loans_Columns_Final.pkl")



#function to transform dependents to integer
def dependents_int(dependents):
    try:
        dependents = int(dependents)
    except:
        dependents = int(dependents.replace('+',''))
    return dependents
    
#function to calculate min_installment
def installment(loan,months):
    return loan *1000 *(1-.2) /months

#function to calculate ratio_of_income
def ratio_income(installment,income):
    return installment /income *100

#function to calculate the rest of whole incomes after minimum installment
#def diff_incomes(ApplicantIncome,CoapplicantIncome,installment):
#    return ApplicantIncome + CoapplicantIncome - installment

 #main 
def main():
    ## Setting up the page title and icon
    st.set_page_config(page_icon = 'loan.png',page_title= 'Loan Eligibility Prediction')
    # Add a title in the middle of the page using Markdown and CSS
    st.markdown("<h1 style='text-align: center;text-decoration: underline;color:GoldenRod'>Loan Eligibility Prediction</h1>", unsafe_allow_html=True)

    #record from user
    
    gender = st.radio('Select Gender' ,['Male','Female'],horizontal=True)
    married = st.radio('Married?',['Yes','No'],horizontal=True)
    dependents = st.selectbox("How many Dependents" ,['0','1','2','3+'])
    Education = st.radio('Education?' ,['Graduate','Not Graduate'],horizontal=True)
    
    Self_Employed = st.radio('Self_Employed?',['Yes','No'],horizontal=True)
    
    Applicant_Income = st.slider('Applicant Income?',150.0 ,81000.0 ,step=1.0)
    
    Coapplicant_Income  = st.slider("Coapplicant Income:",0 , 41667 ,step=1)

    LoanAmount  = st.slider("Loan Amount:(with thouthand)",9.0 , 700.0 , step=1.0)

    Loan_Amount_Term  = st.selectbox('Loan Amount Term',[480.0  ,360.0 ,300.0 ,240.0 ,180.0  ,120.0 ,84.0 ,60.0 ,36.0 ,12.0])
    Credit_History = st.radio('Credit History:',[0,1],horizontal=True)
    Property_Area = st.radio('Property Area:',['Semiurban','Urban','Rural'],horizontal=True)

    #calculate rest features by calling its functions
    min_installment = installment(LoanAmount,Loan_Amount_Term)
    ratio_of_income = ratio_income(min_installment,Applicant_Income)
    #diff = diff_incomes(Applicant_Income,Coapplicant_Income,min_installment)

    

    
#columns:['Gender','Married','Dependents','Education','Self_Employed','ApplicantIncome','CoapplicantIncome','LoanAmount','Loan_Amount_Term','Credit_History','Property_Area',
       #'installment', 'ratio_of_income', 'ratio_of_all_incomes', 'diff','diff_all']

    #create the dataframe of the user's record 
    df =pd.DataFrame(columns=Inputs)
    df.at[0,'Gender']= gender
    df.at[0,'Married']= married
    df.at[0,'Dependents']= dependents_int(dependents)
    df.at[0,'Education']= Education
    df.at[0,'Self_Employed']=  Self_Employed
    df.at[0,'ApplicantIncome']= Applicant_Income
    df.at[0,'CoapplicantIncome']=  Coapplicant_Income
    df.at[0,'LoanAmount']=  LoanAmount
    df.at[0,'Loan_Amount_Term']=  Loan_Amount_Term
    df.at[0,'Credit_History']=  Credit_History
    df.at[0,'Property_Area']=  Property_Area
    df.at[0,'min_installment']=  min_installment
    df.at[0,'ratio_of_income']=  ratio_of_income
    #df.at[0,'diff_all']=  diff


    
    #button to predict
    if st.button('predict'):
        st.dataframe(df)
        result= Model.predict(df)[0]

        
        if result == 1:
            st.success("Loan approved")

        else:
            st.warning("Loan rejected!")

if __name__ == '__main__':
    main()


import pandas as pd
import numpy as np
import streamlit as st
from sklearn.preprocessing import LabelEncoder

df1 = pd.read_csv("I:\\BIke Price Prediction\\Used_Bikes.csv")

df2 = df1.drop(['power', 'brand', 'city'], axis=1)

df2['price'] = df2['price'].astype('int32')
df2['kms_driven'] = df2['kms_driven'].astype('int32')
df2['age'] = df2['age'].astype('int32')
df3 = df2.copy()

df4 = df3.copy()
upper_limit = df4['age'].mean() + 2.3 * df4['age'].std()
df5 = df4.loc[(df4['age'] < upper_limit)]

upper_limit_1 = df5['kms_driven'].mean() + 3.4 * df5['kms_driven'].std()
df6 = df5.loc[(df5['kms_driven'] < upper_limit_1)]

upper_limit_2 = df6['price'].mean() + 0.4 * df6['price'].std()
df7 = df6.loc[(df6['price'] < upper_limit_2)]

le_owner = LabelEncoder()
df7['Owner_Type'] = le_owner.fit_transform(df7['owner'])

df8 = df7.drop(['owner'], axis=1)

dummies = pd.get_dummies(df8['bike_name'])
dummies1 = dummies.drop(['Yamaha YZF-R15 V3 150cc'], axis=1)

df9 = pd.concat([df8, dummies1], axis=1)
df10 = df9.drop(['bike_name'], axis=1)

X = df10.drop(['price'], axis=1)
y = df10['price']

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)

from sklearn import linear_model
model = linear_model.LinearRegression()
model.fit(x_test, y_test)

# Swapped owner types
owner_type_mapping = {
    'First owner': 0,
    'Second owner': 1,
    'Third owner': 2,
    'Fourth owner or more': 3
}

def predict_price(location, kms_driven, age, owner_type):
    loc_index = np.where(X.columns == location)[0][0]

    x = np.zeros(len(X.columns))
    x[0] = kms_driven
    x[1] = age
    x[2] = owner_type
    if loc_index >= 0:
        x[loc_index] = 1

    return model.predict([x])[0]

# Streamlit App
st.title("Used Bike Price Prediction")
st.markdown("<p style='text-align: center; font-size: 16px; color: #808080;'>Made by Arijit Goswami</p>", unsafe_allow_html=True)

# Apply some basic styling using HTML and CSS
st.markdown(
    """
    <style>
        body {
            font-family: 'Arial', sans-serif;
            background-color: #c0c0c0;  /* Change this to your desired background color */
        }
        .title {
            color: #1f78b4;
            text-align: center;
            font-size: 32px;
            margin-bottom: 20px;
        }
        .input-label {
            font-weight: bold;
            margin-top: 10px;
        }
        .prediction {
            color: #4daf4a;
            font-size: 24px;
            margin-top: 20px;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Input form
st.header("Input Features:")
location_options = df9['bike_name'].unique()
location = st.selectbox("Select Bike Model", location_options)
kms_driven = st.number_input("Enter Kilometers Driven", min_value=0)
age = st.number_input("Enter Age of the Bike", min_value=0)
owner_type = st.selectbox("Select Owner Type", list(owner_type_mapping.keys()))

# Convert owner type using mapping
owner_type_encoded = owner_type_mapping[owner_type]

# Predict button
if st.button("Predict Price"):
    try:
        result = predict_price(location, kms_driven, age, owner_type_encoded)
        st.markdown(f"<p class='prediction'>Predicted Price: Rs. {result:.2f}</p>", unsafe_allow_html=True)
    except Exception as e:
        # Handle the exception by showing the price from df1
        selected_bike_price = df1[df1['bike_name'] == location]['price'].values
        if len(selected_bike_price) > 0:
            st.markdown(f"<p class='prediction'>Predicted Price: Rs. {selected_bike_price[0]:.2f}</p>", unsafe_allow_html=True)
        else:
            st.error(f"Error: {str(e)}. Could not find price information for the selected bike.")

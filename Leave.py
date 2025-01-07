

import pandas as pd
import streamlit as st

# Define containers for better layout
header = st.container()
dataset = st.container()
features = st.container()
model_training = st.container()

# Header Section
with header:
    st.title("Welcome to My Awesome Leave Request Application!")
    st.text("In this simple project, we look into the leave calculation for each employee!")

# Dataset Section
with dataset:
    st.header("Dataset Overview")
    st.text("This section can show the dataset or sample data being analyzed.")
    # Placeholder for dataset upload or display
    st.write("Dataset loading and preview can go here.")

    # Load dataset
    mydata = pd.read_csv('employee_leave_data.txt')

    # Display the dataset
    st.write(mydata)

    # Count Leave Taken distribution
    countleave = mydata['Leave_Taken'].value_counts().reset_index()
    countleave.columns = ['Leave_Taken', 'Count']

    # Display bar chart
    st.bar_chart(countleave.set_index('Leave_Taken'))






# import pandas as pd
# import streamlit as st
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# # Define containers for better layout
# header = st.container()
# dataset = st.container()
# features = st.container()
# model_training = st.container()

# # Header Section
# with header:
#     st.title("Welcome to My Awesome Leave Request Application!")
#     st.text("In this simple project, we look into the leave calculation for each employee!")

# # Dataset Section
# with dataset:
#     st.header("Dataset Overview")
#     st.text("This section can show the dataset or sample data being analyzed.")
#     # Placeholder for dataset upload or display
#     st.write("Dataset loading and preview can go here.")

#     # Load dataset
#     mydata = pd.read_csv('employee_leave_data.txt')

#     # Strip leading and trailing spaces from column names
#     mydata.columns = mydata.columns.str.strip()

#     # Convert 'Start_Date' and 'End_Date' to datetime format
#     mydata['Start_Date'] = pd.to_datetime(mydata['Start_Date'], errors='coerce')
#     mydata['End_Date'] = pd.to_datetime(mydata['End_Date'], errors='coerce')

#     # Calculate Leave Duration (difference between start and end dates in days)
#     mydata['Leave_Duration'] = (mydata['End_Date'] - mydata['Start_Date']).dt.days

#     # Display the dataset and column names
#     st.write(mydata.head())
#     st.write("Columns in dataset:", mydata.columns)

#     # Count Leave Taken and Display
#     st.subheader("Leave Taken Distribution")
#     if 'Leave_Taken' in mydata.columns:
#         # Create a DataFrame for leave counts
#         countleave = mydata['Leave_Taken'].value_counts().reset_index()
#         countleave.columns = ['Leave_Taken', 'Count']
        
#         # Display bar chart
#         st.bar_chart(countleave.set_index('Leave_Taken'))
#     else:
#         st.error("Column 'Leave_Taken' not found in the dataset.")

# # Features Section
# with features:
#     st.header("Features Overview")
#     st.text("This section can list the features or metrics being analyzed.")
#     st.write("Examples: Leave Utilization %, Leave Duration Categories, etc.")
#     st.markdown('* **Leave Features:** Leave Start Date, End Date, Duration')
#     st.markdown('* **Leave Utilization %:** Percentage of annual leave used')

#     # Create two columns for side-by-side layout
#     sel_col, disp_col = st.columns(2)

#     max_depth = sel_col.slider('What should be the max-depth of the model?',
#                                min_value=10, max_value=100, value=20, step=10)

#     # Set number of trees in the model
#     n_estimator = sel_col.selectbox('How many trees should the model use?', 
#                                     options=[100, 200, 300, 400, 'NO LIMIT'], 
#                                     index=0)

#     # Input features as a comma-separated string, and split it into a list
#     input_features = sel_col.text_input('Which features should be used? (comma-separated)', 'Total_Annual_Leave, Leave_Remaining, Leave_Duration')
#     input_features = input_features.split(',')
#     input_features = [feature.strip() for feature in input_features]  # Strip spaces from feature names

#     # Check if all features exist in the dataset
#     missing_columns = [col for col in input_features if col not in mydata.columns]
#     if missing_columns:
#         st.error(f"Missing columns in the dataset: {missing_columns}")
#     else:
#         # Initialize the regressor model
#         if n_estimator == 'NO LIMIT':
#             regr = RandomForestRegressor(max_depth=max_depth)
#         else:
#             regr = RandomForestRegressor(max_depth=max_depth, n_estimators=n_estimator)

#         # Selecting the features and target
#         x = mydata[input_features]
#         y = mydata[['Leave_Taken']]

#         # Fitting the model
#         regr.fit(x, y)

#         # Making predictions
#         Prediction = regr.predict(x)
#         st.write("Predictions: ", Prediction)

#         # Display model evaluation metrics
#         st.subheader("Model Evaluation")
#         st.write("Mean Absolute Error:", mean_absolute_error(y, Prediction))
#         st.write("Mean Squared Error:", mean_squared_error(y, Prediction))
#         st.write("R2 Score:", r2_score(y, Prediction))


# import pandas as pd
# import streamlit as st
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# # Define containers for better layout
# header = st.container()
# dataset = st.container()
# features = st.container()
# model_training = st.container()

# # Header Section
# with header:
#     st.title("Welcome to My Awesome Leave Request Application!")
#     st.text("In this simple project, we look into the leave calculation for each employee!")

# # Dataset Section
# with dataset:
#     st.header("Dataset Overview")
#     st.text("This section can show the dataset or sample data being analyzed.")
#     # Load dataset
#     mydata = pd.read_csv('employee_leave_data.txt')

#     # Strip leading and trailing spaces from column names
#     mydata.columns = mydata.columns.str.strip()

#     # Convert 'Start_Date' and 'End_Date' to datetime format
#     mydata['Start_Date'] = pd.to_datetime(mydata['Start_Date'], errors='coerce')
#     mydata['End_Date'] = pd.to_datetime(mydata['End_Date'], errors='coerce')

#     # Calculate Leave Duration (difference between start and end dates in days)
#     mydata['Leave_Duration'] = (mydata['End_Date'] - mydata['Start_Date']).dt.days

#     st.write(mydata.head())

#     # Employee selection dropdown
#     employee_name = st.selectbox('Select an Employee to View Data:', mydata['Employee_Name'].unique())
    
#     # Filter dataset by selected employee
#     selected_employee_data = mydata[mydata['Employee_Name'] == employee_name]
    
#     st.subheader(f"Leave Details for {employee_name}")
#     st.write(selected_employee_data)

#     # Count Leave Taken and Display
#     st.subheader("Leave Taken Distribution")
#     if 'Leave_Taken' in mydata.columns:
#         # Create a DataFrame for leave counts
#         countleave = mydata['Leave_Taken'].value_counts().reset_index()
#         countleave.columns = ['Leave_Taken', 'Count']
#         # Display bar chart (dynamic based on slider)
#         st.bar_chart(countleave.set_index('Leave_Taken'))
#     else:
#         st.error("Column 'Leave_Taken' not found in the dataset.")

# # Features Section
# with features:
#     st.header("Features Overview")
#     st.text("This section can list the features or metrics being analyzed.")
#     st.write("Examples: Leave Utilization %, Leave Duration Categories, etc.")
    
#     max_depth = st.slider('What should be the max-depth of the model?',
#                           min_value=10, max_value=100, value=20, step=10)

#     # Set number of trees in the model
#     n_estimator = st.selectbox('How many trees should the model use?', 
#                                options=[100, 200, 300, 400, 'NO LIMIT'], 
#                                index=0)

#     # Input features as a comma-separated string, and split it into a list
#     input_features = st.text_input('Which features should be used? (comma-separated)', 'Total_Annual_Leave, Leave_Remaining, Leave_Duration')
#     input_features = input_features.split(',')
#     input_features = [feature.strip() for feature in input_features]  # Strip spaces from feature names

#     # Check if all features exist in the dataset
#     missing_columns = [col for col in input_features if col not in mydata.columns]
#     if missing_columns:
#         st.error(f"Missing columns in the dataset: {missing_columns}")
#     else:
#         # Initialize the regressor model
#         if n_estimator == 'NO LIMIT':
#             regr = RandomForestRegressor(max_depth=max_depth)
#         else:
#             regr = RandomForestRegressor(max_depth=max_depth, n_estimators=n_estimator)

#         # Selecting the features and target
#         x = mydata[input_features]
#         y = mydata[['Leave_Taken']]

#         # Fitting the model
#         regr.fit(x, y)

#         # Making predictions
#         Prediction = regr.predict(x)
#         st.write("Predictions: ", Prediction)

#         # Display model evaluation metrics
#         st.subheader("Model Evaluation")
#         st.write("Mean Absolute Error:", mean_absolute_error(y, Prediction))
#         st.write("Mean Squared Error:", mean_squared_error(y, Prediction))
#         st.write("R2 Score:", r2_score(y, Prediction))

#         # Dynamically update graph based on slider (model depth)
#         st.subheader("Updated Leave Taken Distribution Based on Model Depth")
#         countleave = mydata['Leave_Taken'].value_counts().reset_index()
#         countleave.columns = ['Leave_Taken', 'Count']
#         # Update bar chart dynamically with new data based on slider
#         st.bar_chart(countleave.set_index('Leave_Taken'))






# import pandas as pd
# import streamlit as st

# from sklearn.ensemble import RandomForestRegressor
# from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score



# # Define containers for better layout
# header = st.container()
# dataset = st.container()
# features = st.container()
# model_training = st.container()

# # Header Section
# with header:
#     st.title("Welcome to My Awesome Leave Request Application!")
#     st.text("In this simple project, we look into the leave calculation for each employee!")

# # Dataset Section
# with dataset:
#     st.header("Dataset Overview")
#     st.text("This section can show the dataset or sample data being analyzed.")
#     # Placeholder for dataset upload or display
#     st.write("Dataset loading and preview can go here.")

#     mydata = pd.read_csv('employee_leave_data.txt')

#     # Strip leading and trailing spaces from column names
#     mydata.columns = mydata.columns.str.strip()

#      # Add Leave Duration (assuming columns 'Leave_Start_Date' and 'Leave_End_Date' exist in your data)
#     # Ensure dates are in datetime format
#     mydata['Start_Date'] = pd.to_datetime(mydata['Start_Date'], errors='coerce')
#     mydata['End_Date'] = pd.to_datetime(mydata['End_Date'], errors='coerce')

#     # Calculate Leave Duration (difference between start and end dates in days)
#     mydata['Leave_Duration'] = (mydata['End_Date'] - mydata['Start_Date']).dt.days

    
#     st.write(mydata.head())

#    # Count Leave Taken and Display
# st.subheader("Leave Taken Distribution")
# if 'Leave_Taken' in mydata.columns:
#     # Create a DataFrame for leave counts
#     countleave = mydata['Leave_Taken'].value_counts().reset_index()
#     countleave.columns = ['Leave_Taken', 'Count']
    
#     # Display bar chart
#     st.bar_chart(countleave.set_index('Leave_Taken'))
# else:
#     st.error("Column 'Leave_Taken' not found in the dataset.")
# # Features Section

# with features:
#     st.header("Features Overview")
#     st.text("This section can list the features or metrics being analyzed.")
#     st.write("Examples: Leave Utilization %, Leave Duration Categories, etc.")
#     st.markdown('* **first features:** that l created')
#     st.markdown('* **second features:** that l created')

# ## Create two columns for side-by-side layout
#     sel_col, disp_col = st.columns(2)

#     # #In the first column (sel_col), allow user selection or interaction
#     # with sel_col:
#     #     st.subheader("Select Employee")
#     #     employee_name = st.selectbox("Choose an Employee:", mydata['Employee_Name'].unique())
        
#     #     st.subheader("View Data for Selected Employee")
#     #     selected_employee_data = mydata[mydata['Employee_Name'] == employee_name]
#     #     st.write(selected_employee_data)

#     # # In the second column (disp_col), display aggregated or visualized data
#     # with disp_col:
#     #     st.subheader("Leave Taken Distribution")
#     #     countleave = mydata['Leave_Taken'].value_counts().reset_index()
#     #     countleave.columns = ['Leave_Taken', 'Count']
#     #     st.bar_chart(countleave.set_index('Leave_Taken'))

#     max_depth = sel_col.slider('What should be the max-depth of the model?',
#                                min_value=10, max_value=100, value=20, step=10)

#     # Set number of trees in the model
#     n_estimator = sel_col.selectbox('How many trees should the model use?', 
#                                     options=[100, 200, 300, 400, 'NO LIMIT'], 
#                                     index=0)

#     # Input features as a comma-separated string, and split it into a list
#     # Input features as a comma-separated string, and split it into a list
#     input_features = sel_col.text_input('Which features should be used? (comma-separated)', 'Total_Annual_Leave, Leave_Remaining, Leave_Duration')
#     input_features = input_features.split(',')
#     input_features = [feature.strip() for feature in input_features]  # Strip spaces from feature names

#     # Check if all features exist in the dataset
#     missing_columns = [col for col in input_features if col not in mydata.columns]
#     if missing_columns:
#         st.error(f"Missing columns in the dataset: {missing_columns}")
#     else:
#         # Initialize the regressor model
#         if n_estimator == 'NO LIMIT':
#             regr = RandomForestRegressor(max_depth=max_depth)
#         else:
#             regr = RandomForestRegressor(max_depth=max_depth, n_estimators=n_estimator)

#         # Selecting the features and target
#         x = mydata[input_features]
#         y = mydata[['Leave_Taken']]

#         # Fitting the model
#         regr.fit(x, y)

#         # Making predictions
#         Prediction = regr.predict(x)
    

















# import pandas as pd
# import streamlit as st
# from scipy.stats import zscore

# # Load the leave data
# try:
#     df = pd.read_csv('employee_leave_data.txt')
# except FileNotFoundError:
#     st.error("File not found. Please ensure 'employee_leave_data.txt' is in the directory.")
#     st.stop()
# except pd.errors.ParserError:
#     st.error("Error parsing the file. Please check the data format.")
#     st.stop()

# # Data Preparation
# try:
#     df['Start_Date'] = pd.to_datetime(df['Start_Date'], errors='coerce')
#     df['End_Date'] = pd.to_datetime(df['End_Date'], errors='coerce')

#     # Handle missing or invalid dates
#     if df['Start_Date'].isnull().any() or df['End_Date'].isnull().any():
#         st.error("Some rows have invalid dates. Please check the data.")
#         st.stop()

#     df['Leave_Duration'] = (df['End_Date'] - df['Start_Date']).dt.days + 1
#     df['Leave_Utilization_%'] = (df['Leave_Taken'] / df['Total_Annual_Leave']) * 100

#     # Extract Month for Trends
#     df['Month'] = df['Start_Date'].dt.month

#     # Leave duration categories
#     def categorize_duration(days):
#         if days <= 5:
#             return 'Short'
#         elif 6 <= days <= 10:
#             return 'Medium'
#         else:
#             return 'Long'

#     df['Duration_Category'] = df['Leave_Duration'].apply(categorize_duration)

#     # Z-Score for Leave Duration Outliers
#     df['Leave_Duration_Z'] = zscore(df['Leave_Duration'])
#     outliers = df[abs(df['Leave_Duration_Z']) > 2]

# except KeyError as e:
#     st.error(f"Missing required column: {str(e)}")
#     st.stop()

# # Sidebar Filters
# st.sidebar.header("Filters")
# employee_filter = st.sidebar.multiselect("Select Employees", df['Employee_Name'].unique())
# duration_filter = st.sidebar.multiselect("Select Duration Category", ['Short', 'Medium', 'Long'])
# month_filter = st.sidebar.multiselect("Select Month", list(range(1, 13)))

# filtered_df = df.copy()

# # Apply filters
# if employee_filter:
#     filtered_df = filtered_df[filtered_df['Employee_Name'].isin(employee_filter)]

# if duration_filter:
#     filtered_df = filtered_df[filtered_df['Duration_Category'].isin(duration_filter)]

# if month_filter:
#     filtered_df = filtered_df[filtered_df['Month'].isin(month_filter)]

# # Streamlit Layout

# # Header
# st.title("Employee Leave Dashboard")

# # Filtered Data
# st.header("Filtered Data")
# st.write(filtered_df)

# # Total Leave Taken by All Employees
# st.header("Total Leave Taken by Filtered Employees")
# total_leave_taken = filtered_df['Leave_Taken'].sum()
# st.write(f"Total leave taken: {total_leave_taken} days")

# # Leave Remaining for Each Employee
# st.header("Leave Remaining for Filtered Employees")
# leave_remaining = filtered_df[['Employee_Name', 'Leave_Remaining']]
# st.write(leave_remaining)

# # Top 3 Employees Who Took the Most Leave
# st.header("Top 3 Employees with Most Leave Taken (Filtered)")
# top_3_employees = filtered_df.nlargest(3, 'Leave_Taken')[['Employee_Name', 'Leave_Taken', 'Reason']]
# st.write(top_3_employees)

# # Leave Trends by Month
# st.header("Leave Trends by Month (Filtered)")
# leave_trends = filtered_df.groupby('Month').size().reset_index(name='Leave_Count')
# st.bar_chart(leave_trends.set_index('Month')['Leave_Count'])

# # Leave Utilization Percentage
# st.header("Leave Utilization (%)")
# leave_utilization = filtered_df[['Employee_Name', 'Leave_Utilization_%']]
# st.write(leave_utilization)

# # Outliers Based on Leave Duration
# st.header("Leave Duration Outliers (Filtered)")
# filtered_outliers = filtered_df[abs(filtered_df['Leave_Duration_Z']) > 2]
# st.write(filtered_outliers[['Employee_Name', 'Leave_Duration', 'Leave_Duration_Z']])

# # Most Leave Taken (Single Employee)
# st.header("Employee Who Took the Most Leave (Filtered)")
# max_leave_taken = filtered_df[filtered_df['Leave_Taken'] == filtered_df['Leave_Taken'].max()]
# st.write(max_leave_taken[['Employee_Name', 'Leave_Taken']])

# # Longest Leave Taken (Employee)
# st.header("Longest Leave Taken (Filtered)")
# longest_leave = filtered_df[filtered_df['Leave_Duration'] == filtered_df['Leave_Duration'].max()]
# st.write(longest_leave[['Employee_Name', 'Leave_Duration', 'Reason']])

# # Employees Eligible for Leave
# st.header("Employees Eligible for Leave (Filtered)")
# eligible_for_leave = filtered_df[filtered_df['Leave_Remaining'] > 5][['Employee_Name', 'Leave_Remaining']]
# st.write(eligible_for_leave)

# # Leave Duration Categories
# st.header("Leave Duration Categories (Filtered)")
# st.write(filtered_df[['Employee_Name', 'Leave_Duration', 'Duration_Category']])










# import pandas as pd
# import streamlit as st
# from scipy.stats import zscore

# # Load the leave data
# try:
#     df = pd.read_csv('employee_leave_data.txt')
# except FileNotFoundError:
#     st.error("File not found. Please ensure 'employee_leave_data.txt' is in the directory.")
#     st.stop()
# except pd.errors.ParserError:
#     st.error("Error parsing the file. Please check the data format.")
#     st.stop()

# # Data Preparation
# try:
#     df['Start_Date'] = pd.to_datetime(df['Start_Date'], errors='coerce')
#     df['End_Date'] = pd.to_datetime(df['End_Date'], errors='coerce')

#     # Handle missing or invalid dates
#     if df['Start_Date'].isnull().any() or df['End_Date'].isnull().any():
#         st.error("Some rows have invalid dates. Please check the data.")
#         st.stop()

#     df['Leave_Duration'] = (df['End_Date'] - df['Start_Date']).dt.days + 1
#     df['Leave_Utilization_%'] = (df['Leave_Taken'] / df['Total_Annual_Leave']) * 100

#     # Extract Month for Trends
#     df['Month'] = df['Start_Date'].dt.month

#     # Leave duration categories
#     def categorize_duration(days):
#         if days <= 5:
#             return 'Short'
#         elif 6 <= days <= 10:
#             return 'Medium'
#         else:
#             return 'Long'

#     df['Duration_Category'] = df['Leave_Duration'].apply(categorize_duration)

#     # Z-Score for Leave Duration Outliers
#     df['Leave_Duration_Z'] = zscore(df['Leave_Duration'])
#     outliers = df[abs(df['Leave_Duration_Z']) > 2]

# except KeyError as e:
#     st.error(f"Missing required column: {str(e)}")
#     st.stop()

# # Streamlit Layout

# # Header
# st.title("Employee Leave Dashboard")

# # Total Leave Taken by All Employees
# st.header("Total Leave Taken by All Employees")
# total_leave_taken = df['Leave_Taken'].sum()
# st.write(f"Total leave taken: {total_leave_taken} days")

# # Leave Remaining for Each Employee
# st.header("Leave Remaining")
# leave_remaining = df[['Employee_Name', 'Leave_Remaining']]
# st.write(leave_remaining)

# # Top 3 Employees Who Took the Most Leave
# st.header("Top 3 Employees with Most Leave Taken")
# top_3_employees = df.nlargest(3, 'Leave_Taken')[['Employee_Name', 'Leave_Taken', 'Reason']]
# st.write(top_3_employees)

# # Leave Trends by Month
# st.header("Leave Trends by Month")
# leave_trends = df.groupby('Month').size().reset_index(name='Leave_Count')
# st.bar_chart(leave_trends.set_index('Month')['Leave_Count'])

# # Leave Utilization Percentage
# st.header("Leave Utilization (%)")
# leave_utilization = df[['Employee_Name', 'Leave_Utilization_%']]
# st.write(leave_utilization)

# # Outliers Based on Leave Duration
# st.header("Leave Duration Outliers")
# st.write(outliers[['Employee_Name', 'Leave_Duration', 'Leave_Duration_Z']])

# # Most Leave Taken (Single Employee)
# st.header("Employee Who Took the Most Leave")
# max_leave_taken = df[df['Leave_Taken'] == df['Leave_Taken'].max()]
# st.write(max_leave_taken[['Employee_Name', 'Leave_Taken']])

# # Longest Leave Taken (Employee)
# st.header("Longest Leave Taken")
# longest_leave = df[df['Leave_Duration'] == df['Leave_Duration'].max()]
# st.write(longest_leave[['Employee_Name', 'Leave_Duration', 'Reason']])

# # Employees Eligible for Leave
# st.header("Employees Eligible for Leave")
# eligible_for_leave = df[df['Leave_Remaining'] > 5][['Employee_Name', 'Leave_Remaining']]
# st.write(eligible_for_leave)

# # Leave Duration Categories
# st.header("Leave Duration Categories")
# st.write(df[['Employee_Name', 'Leave_Duration', 'Duration_Category']])

# # Debugging Logs
# if 'Month' in df.columns:
#     st.write("Month column created successfully.")
# else:
#     st.error("Failed to create the Month column.")








# import os
# import pandas as pd
# import streamlit as st
# import plotly.express as px

# # Title and Description
# st.title("Avocado Prices Dashboard")
# st.markdown("""
# This is an interactive dashboard displaying **average avocado prices** across different geographies.  
# Data Source: [Kaggle](https://www.kaggle.com/datasets/timmate/avocado-prices-2020)  
# Use the dropdown menu to select a city and view the corresponding trends.
# """)

# # File Path
# file_path = "avocado.csv"

# # Check if the file exists or use a sample dataset
# if os.path.exists(file_path):
#     avocado = pd.read_csv(file_path)
#     st.success("Data successfully loaded!")
# else:
#     st.warning("Dataset not found. Using sample data for demonstration purposes.")
#     data = {
#         'date': ['2020-01-01', '2020-02-01', '2020-03-01', '2020-01-01', '2020-02-01', '2020-03-01'],
#         'geography': ['Los Angeles', 'Los Angeles', 'Los Angeles', 'New York', 'New York', 'New York'],
#         'type': ['conventional', 'organic', 'conventional', 'organic', 'conventional', 'organic'],
#         'average_price': [1.2, 1.5, 1.3, 1.8, 1.9, 2.0]
#     }
#     avocado = pd.DataFrame(data)

# # Ensure date column is in datetime format
# avocado['date'] = pd.to_datetime(avocado['date'])

# # Display Dataset
# st.header("Dataset Overview")
# st.write(avocado.head())

# # Summary Statistics
# st.header("Summary Statistics")
# st.write(avocado.describe())

# # Line Chart: User Selection
# st.header("Interactive Line Chart")
# selected_geo = st.selectbox("Select a Geography:", avocado['geography'].unique())
# filtered_data = avocado[avocado['geography'] == selected_geo]

# # Line Chart
# line_fig = px.line(
#     filtered_data,
#     x='date',
#     y='average_price',
#     color='type',
#     title=f"Average Avocado Prices in {selected_geo}"
# )
# st.plotly_chart(line_fig)

# # Footer
# st.markdown("""
# ---
# *Developed with Streamlit & Plotly Express*  
# :avocado: **Happy Visualizing!**
# """)








# import streamlit as st

# # Title for the Streamlit app
# st.title("My First Streamlit App")

# # Welcome message
# st.write("Welcome to my Streamlit App")

# # Text input for a custom message
# user_input = st.text_input("Enter a custom message:", "Hello Adeniyi")

# # Display the customized message
# st.write("Customized message:", user_input)








# import streamlit as st
# import pandas as pd
# import plotly.graph_objects as go

# # Dashboard title and description
# st.title("Stroke Prediction Dashboard")
# st.markdown("This dashboard helps researchers explore the dataset and analyze outcomes.")

# # Sidebar options
# st.sidebar.title("Select Visual Charts")
# st.sidebar.markdown("Choose the type of chart/plot:")

# # Load dataset
# try:
#     data = pd.read_csv("PyCharm_Projects/Streamlit/demo_data_set.csv")
# except FileNotFoundError:
#     st.error("Dataset not found. Please check the file path.")
#     st.stop()

# # Sidebar selections
# chart_visual = st.sidebar.selectbox("Select Chart Type", ['Line Chart', 'Bar Chart', 'Bubble Chart'])
# smoking_status = st.sidebar.selectbox("Select Smoking Status", 
#                                       ['Formerly_Smoked', 'Smoked', 'Never_Smoked', 'Unknown'])

# # Plotly figure
# fig = go.Figure()

# # Add trace based on user selection
# status_column_map = {
#     'Formerly_Smoked': 'formerly_smoked',
#     'Smoked': 'Smokes',
#     'Never_Smoked': 'Never_Smoked',
#     'Unknown': 'Unknown'
# }

# if smoking_status in status_column_map:
#     y_column = status_column_map[smoking_status]
#     if y_column not in data.columns:
#         st.error(f"Column '{y_column}' not found in the dataset.")
#         st.stop()

#     x_data = data['Country']  # Ensure 'Country' exists in dataset
#     y_data = data[y_column]

#     if chart_visual == 'Line Chart':
#         fig.add_trace(go.Scatter(x=x_data, y=y_data, mode='lines', name=smoking_status))
#     elif chart_visual == 'Bar Chart':
#         fig.add_trace(go.Bar(x=x_data, y=y_data, name=smoking_status))
#     elif chart_visual == 'Bubble Chart':
#         # Generate marker sizes dynamically
#         marker_size = [val * 10 for val in y_data] if not y_data.isnull().all() else [10] * len(y_data)
#         fig.add_trace(go.Scatter(x=x_data, y=y_data, mode='markers', 
#                                  marker=dict(size=marker_size), name=smoking_status))

# # Display chart
# st.plotly_chart(fig, use_container_width=True)










# import streamlit as stm 
# from streamlit_card import card 


# stm.set_page_config(page_title="This is a Simple Streamlit WebApp") 
# stm.title("This is the Home Page Geeks.") 
# stm.text("Geeks Home Page") 


# # Card 

# card( 
# 	title="Hello Geeks!", 
# 	text="Click this card to redirect to GeeksforGeeks", 
# 	image="https://media.geeksforgeeks.org/wp-content/cdn-uploads/20190710102234/download3.png", 
# 	url="https://www.geeksforgeeks.org/", 
# ) 





# import streamlit as stm 
# from streamlit_extras.buy_me_a_coffee import button 

# stm.set_page_config(page_title = "This is a Simple Streamlit WebApp") 
# stm.title("This is the Home Page Geeks.") 
# stm.text("Geeks Home Page") 

# button(username="Geeks", floating=False, width=250)





# # import module
# import streamlit as st

# # Title
# st.title("Hello GeeksForGeeks !!!")

# # Header
# st.header("This is a header") 

# # Subheader
# st.subheader("This is a subheader")

# # Text
# st.text("Hello GeeksForGeeks!!!")

# # Markdown
# st.markdown("### This is a markdown")

# # success
# st.success("Success")

# # success
# st.info("Information")

# # success
# st.warning("Warning")

# # success
# st.error("Error")

# # Exception - This has been added later
# exp = ZeroDivisionError("Trying to divide by Zero")
# st.exception(exp)


# # Write text
# st.write("Text with write")

# # Writing python inbuilt function range()
# st.write(range(10))

# # Display Images

# # import Image from pillow to open images
# from PIL import Image
# img = Image.open("bet9ja.png")

# # display image using streamlit
# # width is used to set the width of an image
# st.image(img, width=200)


# # checkbox
# # check if the checkbox is checked
# # title of the checkbox is 'Show/Hide'
# if st.checkbox("Show/Hide"):

#     # display the text if the checkbox returns True value
#     st.text("Showing the widget")

# # radio button
# # first argument is the title of the radio button
# # second argument is the options for the radio button
# status = st.radio("Select Gender: ", ('Male', 'Female'))

# # conditional statement to print 
# # Male if male is selected else print female
# # show the result using the success function
# if (status == 'Male'):
#     st.success("Male")
# else:
#     st.success("Female")
                
# # Selection box

# # first argument takes the titleof the selectionbox
# # second argument takes options
# hobby = st.selectbox("Hobbies: ",
#                      ['Dancing', 'Reading', 'Sports'])

# # print the selected hobby
# st.write("Your hobby is: ", hobby)


# # multi select box

# # first argument takes the box title
# # second argument takes the options to show
# hobbies = st.multiselect("Hobbies: ",
#                          ['Dancing', 'Reading', 'Sports'])

# # write the selected options
# st.write("You selected", len(hobbies), 'hobbies')



# # Create a simple button that does nothing
# st.button("Click me for no reason")

# # Create a button, that when clicked, shows a text
# if(st.button("About")):
#     st.text("Welcome To GeeksForGeeks!!!")




# # Create a text input box with a default placeholder
# name = st.text_input("Enter Your Name", "Type Here ...")

# # When the submit button is clicked, process and display the name
# if st.button('Submit'):
#     # Capitalize the first letter of each word in the input
#     result = name.upper()
#     # Display the result as a success message
#     st.success(f"Hello, {result}!")





# # slider

# # first argument takes the title of the slider
# # second argument takes the starting of the slider
# # last argument takes the end number
# level = st.slider("Select the level", 1, 5)

# # print the level
# # format() is used to print value 
# # of a variable at a specific position
# st.text('Selected: {}'.format(level))






# # give a title to our app
# st.title('Welcome to BMI Calculator')

# # TAKE WEIGHT INPUT in kgs
# weight = st.number_input("Enter your weight (in kgs)")

# # TAKE HEIGHT INPUT
# # radio button to choose height format
# status = st.radio('Select your height format: ',
#                   ('cms', 'meters', 'feet'))

# # compare status value
# if(status == 'cms'):
#     # take height input in centimeters
#     height = st.number_input('Centimeters')

#     try:
#         bmi = weight / ((height/100)**2)
#     except:
#         st.text("Enter some value of height")

# elif(status == 'meters'):
#     # take height input in meters
#     height = st.number_input('Meters')

#     try:
#         bmi = weight / (height ** 2)
#     except:
#         st.text("Enter some value of height")

# else:
#     # take height input in feet
#     height = st.number_input('Feet')

#     # 1 meter = 3.28
#     try:
#         bmi = weight / (((height/3.28))**2)
#     except:
#         st.text("Enter some value of height")

# # check if the button is pressed or not
# if(st.button('Calculate BMI')):

#     # print the BMI INDEX
#     st.text("Your BMI Index is {}.".format(bmi))

#     # give the interpretation of BMI index
#     if(bmi < 16):
#         st.error("You are Extremely Underweight")
#     elif(bmi >= 16 and bmi < 18.5):
#         st.warning("You are Underweight")
#     elif(bmi >= 18.5 and bmi < 25):
#         st.success("Healthy")
#     elif(bmi >= 25 and bmi < 30):
#         st.warning("Overweight")
#     elif(bmi >= 30):
#         st.error("Extremely Overweight")















# import pandas as pd
# from scipy.stats import zscore
# import calendar

# # Step 1: Creating and writing employee leave data (with additional leave types) to a file
# file_path = 'employee_leave_data.txt'
# with open(file_path, 'w') as fp:
#     fp.write('Employee_ID,Employee_Name,Start_Date,End_Date,Leave_Taken,Total_Annual_Leave,Leave_Remaining,Reason\n')
#     fp.write('E101,John Doe,2024-01-05,2024-01-10,5,20,15,Annual Leave\n')
#     fp.write('E102,Jane Smith,2024-02-01,2024-02-03,2,20,18,Personal Leave\n')
#     fp.write('E103,Mike Johnson,2024-03-15,2024-03-20,5,20,15,Sick Leave\n')
#     fp.write('E104,Emily Davis,2024-04-05,2024-04-07,2,20,18,Casual Leave\n')
#     fp.write('E105,James Wilson,2024-05-10,2024-05-14,4,20,16,Annual Leave\n')
#     fp.write('E106,Sarah Brown,2024-06-01,2024-06-30,30,20,-10,Maternity Leave\n')
#     fp.write('E107,Robert Green,2024-07-01,2024-07-05,5,20,15,Paternity Leave\n')
#     fp.write('E108,Linda White,2024-08-10,2024-08-20,10,20,10,Compassionate Leave\n')
#     fp.write('E109,David Black,2024-09-15,2024-09-25,10,20,10,Study Leave\n')
# print(f"Employee leave data has been written to '{file_path}'.")

# # Step 2: Reading the employee leave data into a pandas DataFrame
# df = pd.read_csv(file_path)

# # Convert dates to datetime
# df['Start_Date'] = pd.to_datetime(df['Start_Date'])
# df['End_Date'] = pd.to_datetime(df['End_Date'])

# # Calculate leave duration
# df['Leave_Duration'] = (df['End_Date'] - df['Start_Date']).dt.days + 1

# # Step 3: Performing initial and additional analyses
# df['Leave_Utilization_%'] = (df['Leave_Taken'] / df['Total_Annual_Leave']) * 100

# # Leave duration categories
# def categorize_duration(days):
#     if days <= 5:
#         return 'Short'
#     elif days >= 6 and days <= 10:

#         return 'Medium'
#     else:
#         return 'Long'

# df['Duration_Category'] = df['Leave_Duration'].apply(categorize_duration)

# # Top 3 employees who have taken the most leave
# top_3_employees = df.nlargest(3, 'Leave_Taken')[['Employee_Name', 'Leave_Taken', 'Reason']]


# # least 3 employees who have taken the least leave
# least_3_employees = df.nsmallest(3, 'Leave_Taken')[['Employee_Name', 'Leave_Taken', 'Reason']]


# # Add a column for the month
# df['Month'] = df['Start_Date'].dt.month

# # Find the month with the most leave requests
# most_leave_month = df['Month'].value_counts().idxmax()

# # Convert the month number to the month name
# most_leave_month_name = calendar.month_name[most_leave_month]



# # Add a column for the month
# df['Month'] = df['Start_Date'].dt.month

# # Find the month with the most leave requests
# least_leave_month = df['Month'].value_counts().idxmax()

# # Convert the month number to the month name
# least_leave_month_name = calendar.month_name[least_leave_month]


# # Employees eligible for leave (leave remaining > threshold)
# eligible_for_leave = df[df['Leave_Remaining'] > 5][['Employee_Name', 'Leave_Remaining']]

# # Step 4: Writing all results to a single file
# output_path = 'leave_analysis_results.txt'
# with open(output_path, 'w') as out:
#     # Total leave taken by all employees
#     total_leave_taken = df['Leave_Taken'].sum()
#     out.write(f"Total Leave Taken by All Employees: {total_leave_taken} days\n\n")

#     # Leave remaining for each employee
#     # df['Leave_Remaining'] = df['Total_Annual_Leave'] - df['Leave_Taken']
#     # leave_remaining = df[['Employee_Name', 'Leave_Remaining']]
#     # out.write("Leave Remaining for Each Employee:\n")
#     # out.write(leave_remaining.to_string(index=False) + "\n\n")


#     # Calculate leave remaining
#     df['Leave_Remaining'] = df['Total_Annual_Leave'] - df['Leave_Taken']

#     #Select the required columns
#     employee_leave_summary = df[['Employee_Name', 'Leave_Taken', 'Leave_Remaining', 'Reason']]

#     out.write("Leave Summary for Each Employee:\n")
#     out.write(employee_leave_summary.to_string(index=False) + "\n\n")


#     # Employee(s) who have taken the most leave
#     max_leave_taken = df[df['Leave_Taken'] == df['Leave_Taken'].max()]
#     out.write("Employee(s) Who Have Taken the Most Leave:\n")
#     out.write(max_leave_taken[['Employee_Name', 'Leave_Taken']].to_string(index=False) + "\n\n")



#     # Employee(s) who have taken the most leave
#     min_leave_taken = df[df['Leave_Taken'] == df['Leave_Taken'].min()]
#     out.write("Employee(s) Who Have Taken the least Leave:\n")
#     out.write(min_leave_taken[['Employee_Name', 'Leave_Taken']].to_string(index=False) + "\n\n")

#     # Leave grouped by reason
#     # grouped_leave_reason = df.groupby('Reason')['Leave_Taken'].sum().reset_index()
#     # out.write("Leave Taken Grouped by Reason:\n")
#     # out.write(grouped_leave_reason.to_string(index=False) + "\n\n")

#     grouped_leave_reason = (
#     df.groupby('Reason')
#     .agg(
#         Leave_Taken=('Leave_Taken', 'sum'),
#         Employee_Name=('Employee_Name', lambda x: ', '.join(x))
#     )
#     .reset_index()
#     )

#     out.write("Leave Taken Grouped by Reason:\n")
#     out.write(grouped_leave_reason.to_string(index=False) + "\n\n")

#     # Employees with negative leave balances
#     negative_balance = df[df['Leave_Remaining'] < 0]
#     out.write("Employees with Negative Leave Balances:\n")
#     out.write(negative_balance[['Employee_Name', 'Leave_Taken', 'Leave_Remaining', 'Reason']].to_string(index=False) + "\n\n")

#     # Leave trends by month
#     leave_trends = df.groupby('Month').size().reset_index(name='Leave_Count')
#     out.write("Leave Trends by Month:\n")
#     out.write(leave_trends.to_string(index=False) + "\n\n")

#     # Employees who have used all their leave
#     all_leave_used = df[df['Leave_Remaining'] == 0]
#     out.write("Employees Who Have Used All Their Leave:\n")
#     out.write(all_leave_used[['Employee_Name', 'Leave_Taken', 'Leave_Remaining', 'Reason']].to_string(index=False) + "\n\n")

#     # Additional Analysis
#     avg_leave_taken = df['Leave_Taken'].mean()
#     out.write(f"Average Leave Taken by Employees: {avg_leave_taken:.2f} days\n\n")

#     # Employees exceeding the leave threshold
#     leave_threshold = 25
#     exceeding_threshold = df[df['Leave_Taken'] > leave_threshold]
#     out.write("Employees Exceeding the Leave Threshold:\n")
#     out.write(exceeding_threshold[['Employee_Name', 'Leave_Taken', 'Reason']].to_string(index=False) + "\n\n")

#     # Longest continuous leave
#     longest_leave = df[df['Leave_Duration'] == df['Leave_Duration'].max()]
#     out.write("Longest Continuous Leave:\n")
#     out.write(longest_leave[['Employee_Name', 'Leave_Duration', 'Reason']].to_string(index=False) + "\n\n")

#     # Leave utilization percentage
#     out.write("Leave Utilization Percentage for Each Employee:\n")
#     utilization = df[['Employee_Name', 'Leave_Utilization_%']]
#     out.write(utilization.to_string(index=False) + "\n\n")

#     # Leave duration outliers using Z-score
#     df['Leave_Duration_Z'] = zscore(df['Leave_Duration'])
#     outliers = df[abs(df['Leave_Duration_Z']) > 2]
#     out.write("Leave Duration Outliers (Z-Score Method):\n")
#     out.write(outliers[['Employee_Name', 'Leave_Duration', 'Leave_Duration_Z']].to_string(index=False) + "\n\n")

#     # Top 3 employees with the most leave
#     out.write("Top 3 Employees Who Have Taken the Most Leave:\n")
#     out.write(top_3_employees.to_string(index=False) + "\n\n")


#     # Top 3 employees with the most leave
#     out.write("least 3 Employees Who Have Taken the least Leave:\n")
#     out.write(least_3_employees.to_string(index=False) + "\n\n")


#     # Month with the most leave requests
#     out.write(f"Month with the Most Leave Requests: {most_leave_month_name} \n\n")


#     # Month with the least leave requests
#     out.write(f"Month with the least Leave Requests: {least_leave_month_name} \n\n")


#     # Leave duration categories
#     out.write("Leave Duration Categorization:\n")
#     categorized_durations = df[['Employee_Name', 'Leave_Duration', 'Duration_Category']]
#     out.write(categorized_durations.to_string(index=False) + "\n\n")

#     # Employees eligible for leave
#     out.write("Employees Eligible for Leave (Based on Leave Remaining):\n")
#     out.write(eligible_for_leave.to_string(index=False) + "\n\n")

# print(f"Leave analysis completed. Results saved to '{output_path}'.")
















# import pandas as pd
# from scipy.stats import zscore
# import calendar

# # Step 1: Creating and writing employee leave data (with additional leave types) to a file
# file_path = 'employee_leave_data.txt'
# with open(file_path, 'w') as fp:
#     fp.write('Employee_ID,Employee_Name,Start_Date,End_Date,Leave_Taken,Total_Annual_Leave,Leave_Remaining,Reason\n')
#     fp.write('E101,John Doe,2024-01-05,2024-01-10,5,20,15,Annual Leave\n')
#     fp.write('E101,John Doe,2024-01-16,2024-01-20,5,20,15,Casual Leave\n')
#     fp.write('E102,Jane Smith,2024-02-01,2024-02-03,2,20,18,Personal Leave\n')
#     fp.write('E103,Mike Johnson,2024-03-15,2024-03-20,5,20,15,Sick Leave\n')
#     fp.write('E104,Emily Davis,2024-04-05,2024-04-07,2,20,18,Casual Leave\n')
#     fp.write('E105,James Wilson,2024-01-10,2024-05-14,4,20,16,Annual Leave\n')
#     fp.write('E106,Sarah Brown,2024-06-01,2024-06-30,30,20,-10,Maternity Leave\n')
#     fp.write('E107,Robert Green,2024-07-01,2024-07-05,5,20,15,Paternity Leave\n')
#     fp.write('E108,Linda White,2024-08-10,2024-08-20,10,20,10,Compassionate Leave\n')
#     fp.write('E109,David Black,2024-09-15,2024-09-25,10,20,10,Study Leave\n')
# print(f"Employee leave data has been written to '{file_path}'.")

# # Step 2: Reading the employee leave data into a pandas DataFrame
# df = pd.read_csv(file_path)

# # Convert dates to datetime
# df['Start_Date'] = pd.to_datetime(df['Start_Date'])
# df['End_Date'] = pd.to_datetime(df['End_Date'])

# # Calculate leave duration
# df['Leave_Duration'] = (df['End_Date'] - df['Start_Date']).dt.days + 1

# # Step 3: Performing initial and additional analyses
# df['Leave_Utilization_%'] = (df['Leave_Taken'] / df['Total_Annual_Leave']) * 100

# # Employees who have taken 2 or more days of leave
# employees_two_or_more_leaves = df[df['Leave_Taken'] >= 2][['Employee_Name', 'Reason', 'Leave_Taken']]


# # Leave duration categories
# def categorize_duration(days):
#     if days <= 5:
#         return 'Short'
#     elif 6 <= days <= 10:
#         return 'Medium'
#     else:
#         return 'Long'

# df['Duration_Category'] = df['Leave_Duration'].apply(categorize_duration)

# # Top 3 employees who have taken the most leave
# top_3_employees = df.nlargest(3, 'Leave_Taken')[['Employee_Name', 'Leave_Taken', 'Reason']]


# # least 3 employees who have taken the least leave
# least_3_employees = df.nsmallest(3, 'Leave_Taken')[['Employee_Name', 'Leave_Taken', 'Reason']]


# # Add a column for the month
# df['Month'] = df['Start_Date'].dt.month

# # Find the month with the most leave requests
# most_leave_month = df['Month'].value_counts().idxmax()

# # Convert the month number to the month name
# most_leave_month_name = calendar.month_name[most_leave_month]

# # print(f"The month with the most leave requests is: {most_leave_month_name}")



# # Employees eligible for leave (leave remaining > threshold)
# eligible_for_leave = df[df['Leave_Remaining'] > 5][['Employee_Name', 'Leave_Remaining']]

# # Step 4: Writing all results to a single file
# output_path = 'leave_analysis_results.txt'
# with open(output_path, 'w') as out:



#     # Total leave taken by all employees
#     total_leave_taken = df['Leave_Taken'].sum()
#     out.write(f"Total Leave Taken by All Employees: {total_leave_taken} days\n\n")


# # Calculate leave remaining
# df['Leave_Remaining'] = df['Total_Annual_Leave'] - df['Leave_Taken']

# # Select the required columns
# employee_leave_summary = df[['Employee_Name', 'Leave_Taken', 'Leave_Remaining', 'Reason']]

# # Write the summary to the output file

#     out.write("Overall leave taken by all Employee:\n")
#     out.write(f"Total Leave Taken by All Employees: {total_leave_taken} days\n\n")


#     out.write("Leave Summary for Each Employee:\n")
#     out.write(employee_leave_summary.to_string(index=False) + "\n\n")



#     # Employee(s) who have taken the most leave
#     max_leave_taken = df[df['Leave_Taken'] == df['Leave_Taken'].max()]
#     out.write("Employee(s) Who Have Taken the Most Leave:\n")
#     out.write(max_leave_taken[['Employee_Name', 'Leave_Taken']].to_string(index=False) + "\n\n")

#     # Employee(s) who have taken the least leave
#     min_leave_taken = df[df['Leave_Taken'] == df['Leave_Taken'].min()]
#     out.write("Employee(s) Who Have Taken the least Leave:\n")
#     out.write(min_leave_taken[['Employee_Name', 'Leave_Taken']].to_string(index=False) + "\n\n")

#     # Leave grouped by reason
#     grouped_leave_reason = df.groupby('Reason')['Leave_Taken'].sum().reset_index()
#     out.write("Leave Taken Grouped by Reason:\n")
#     out.write(grouped_leave_reason.to_string(index=False) + "\n\n")

#     # Employees with negative leave balances
#     negative_balance = df[df['Leave_Remaining'] < 0]
#     out.write("Employees with Negative Leave Balances:\n")
#     out.write(negative_balance[['Employee_Name', 'Leave_Taken', 'Leave_Remaining', 'Reason']].to_string(index=False) + "\n\n")

#     # Leave trends by month
#     leave_trends = df.groupby('Month').size().reset_index(name='Leave_Count')
#     out.write("Leave Trends by Month:\n")
#     out.write(leave_trends.to_string(index=False) + "\n\n")

#     # Employees who have used all their leave
#     all_leave_used = df[df['Leave_Remaining'] == 0]
#     out.write("Employees Who Have Used All Their Leave:\n")
#     out.write(all_leave_used[['Employee_Name', 'Leave_Taken', 'Leave_Remaining', 'Reason']].to_string(index=False) + "\n\n")

#     # Additional Analysis
#     avg_leave_taken = df['Leave_Taken'].mean()
#     out.write(f"Average Leave Taken by Employees: {avg_leave_taken:.2f} days\n\n")

#     # Employees exceeding the leave threshold
#     leave_threshold = 25
#     exceeding_threshold = df[df['Leave_Taken'] > leave_threshold]
#     out.write("Employees Exceeding the Leave Threshold:\n")
#     out.write(exceeding_threshold[['Employee_Name', 'Leave_Taken', 'Reason']].to_string(index=False) + "\n\n")

#     # Longest continuous leave
#     longest_leave = df[df['Leave_Duration'] == df['Leave_Duration'].max()]
#     out.write("Longest Continuous Leave:\n")
#     out.write(longest_leave[['Employee_Name', 'Leave_Duration', 'Reason']].to_string(index=False) + "\n\n")

#     # Leave utilization percentage
#     out.write("Leave Utilization Percentage for Each Employee:\n")
#     utilization = df[['Employee_Name', 'Leave_Utilization_%']]
#     out.write(utilization.to_string(index=False) + "\n\n")

#     # Leave duration outliers using Z-score
#     df['Leave_Duration_Z'] = zscore(df['Leave_Duration'])
#     outliers = df[abs(df['Leave_Duration_Z']) > 2]
#     out.write("Leave Duration Outliers (Z-Score Method):\n")
#     out.write(outliers[['Employee_Name', 'Leave_Duration', 'Leave_Duration_Z']].to_string(index=False) + "\n\n")

#     # Top 3 employees with the most leave
#     out.write("Top 3 Employees Who Have Taken the Most Leave:\n")
#     out.write(top_3_employees.to_string(index=False) + "\n\n")

#      # least 3 employees with the least leave
#     out.write("least 3 Employees Who Have Taken the least Leave:\n")
#     out.write(least_3_employees.to_string(index=False) + "\n\n")

#     # Month with the most leave requests
#     out.write(f"Month with the Most Leave Requests: {most_leave_month_name}\n\n")

#     # Leave duration categories
#     out.write("Leave Duration Categorization:\n")
#     categorized_durations = df[['Employee_Name', 'Leave_Duration', 'Duration_Category']]
#     out.write(categorized_durations.to_string(index=False) + "\n\n")

#     # Employees eligible for leave
#     out.write("Employees Eligible for Leave (Based on Leave Remaining):\n")
#     out.write(eligible_for_leave.to_string(index=False) + "\n\n")


#     # Employees who have taken 2 or more days of leave
#     out.write("Employees Who Have Taken 2 or More Days of Leave:\n")
#     out.write(employees_two_or_more_leaves.to_string(index=False) + "\n")

# print(f"Leave analysis completed. Results saved to '{output_path}'.")









# import pandas as pd
# from scipy.stats import zscore

# # Step 1: Creating and writing employee leave data to an Excel file
# file_path = 'employee_leave_data.xlsx'

# # Creating a DataFrame with the leave data
# leave_data = {
#     'Employee_ID': ['E101', 'E102', 'E103', 'E104', 'E105', 'E106', 'E107', 'E108', 'E109'],
#     'Employee_Name': ['John Doe', 'Jane Smith', 'Mike Johnson', 'Emily Davis', 'James Wilson', 
#                       'Sarah Brown', 'Robert Green', 'Linda White', 'David Black'],
#     'Start_Date': ['2024-01-05', '2024-02-01', '2024-03-15', '2024-04-05', '2024-05-10', 
#                    '2024-06-01', '2024-07-01', '2024-08-10', '2024-09-15'],
#     'End_Date': ['2024-01-10', '2024-02-03', '2024-03-20', '2024-04-07', '2024-05-14', 
#                  '2024-06-30', '2024-07-05', '2024-08-20', '2024-09-25'],
#     'Leave_Taken': [5, 2, 5, 2, 4, 30, 5, 10, 10],
#     'Total_Annual_Leave': [20, 20, 20, 20, 20, 20, 20, 20, 20],
#     'Leave_Remaining': [15, 18, 15, 18, 16, -10, 15, 10, 10],
#     'Reason': ['Annual Leave', 'Personal Leave', 'Sick Leave', 'Casual Leave', 'Annual Leave', 
#                'Maternity Leave', 'Paternity Leave', 'Compassionate Leave', 'Study Leave']
# }

# df = pd.DataFrame(leave_data)

# # Save the data to the Excel file
# df.to_excel(file_path, index=False)

# # Step 2: Reading the employee leave data from the Excel file
# df = pd.read_excel(file_path)

# # Convert dates to datetime
# df['Start_Date'] = pd.to_datetime(df['Start_Date'])
# df['End_Date'] = pd.to_datetime(df['End_Date'])

# # Calculate leave duration
# df['Leave_Duration'] = (df['End_Date'] - df['Start_Date']).dt.days + 1

# # Step 3: Performing analyses
# df['Leave_Utilization_%'] = (df['Leave_Taken'] / df['Total_Annual_Leave']) * 100

# # Employees who have taken 2 or more days of leave
# employees_two_or_more_leaves = df[df['Leave_Taken'] >= 2][['Employee_Name', 'Reason', 'Leave_Taken']]

# # Step 4: Writing all results to a single Excel file
# output_path = 'leave_analysis_results.xlsx'
# with pd.ExcelWriter(output_path) as writer:
#     # Total leave taken by all employees
#     total_leave_taken = df['Leave_Taken'].sum()
#     pd.DataFrame({'Total Leave Taken by All Employees': [total_leave_taken]}).to_excel(writer, sheet_name='Summary', index=False)

#     # Leave remaining for each employee
#     leave_remaining = df[['Employee_Name', 'Leave_Remaining']]
#     leave_remaining.to_excel(writer, sheet_name='Leave Remaining', index=False)

#     # Employee(s) who have taken the most leave
#     max_leave_taken = df[df['Leave_Taken'] == df['Leave_Taken'].max()]
#     max_leave_taken[['Employee_Name', 'Leave_Taken']].to_excel(writer, sheet_name='Most Leave Taken', index=False)

#     # Leave grouped by reason
#     grouped_leave_reason = df.groupby('Reason')['Leave_Taken'].sum().reset_index()
#     grouped_leave_reason.to_excel(writer, sheet_name='Leave by Reason', index=False)

#     # Employees with negative leave balances
#     negative_balance = df[df['Leave_Remaining'] < 0]
#     negative_balance[['Employee_Name', 'Leave_Taken', 'Leave_Remaining', 'Reason']].to_excel(writer, sheet_name='Negative Balances', index=False)

#     # Employees who have taken 2 or more days of leave
#     employees_two_or_more_leaves.to_excel(writer, sheet_name='Two or More Leaves', index=False)

# print(f"Leave analysis completed. Results saved to '{output_path}'.")














# import pandas as pd
# from scipy.stats import zscore

# # Step 1: Creating and writing employee leave data (with department and additional details) to a file
# file_path = 'employee_leave_data.txt'
# with open(file_path, 'w') as fp:
#     fp.write('Employee_ID,Employee_Name,Department,Start_Date,End_Date,Leave_Taken,Total_Annual_Leave,Leave_Remaining,Reason\n')
#     fp.write('E101,John Doe,Finance,2024-01-05,2024-01-10,5,20,15,Annual Leave\n')
#     fp.write('E102,Jane Smith,HR,2024-02-01,2024-02-03,2,20,18,Personal Leave\n')
#     fp.write('E103,Mike Johnson,IT,2024-03-15,2024-03-20,5,20,15,Sick Leave\n')
#     fp.write('E104,Emily Davis,Finance,2024-04-05,2024-04-07,2,20,18,Casual Leave\n')
#     fp.write('E105,James Wilson,Marketing,2024-05-10,2024-05-14,4,20,16,Annual Leave\n')
#     fp.write('E106,Sarah Brown,HR,2024-06-01,2024-06-30,30,20,-10,Maternity Leave\n')
#     fp.write('E107,Robert Green,IT,2024-07-01,2024-07-05,5,20,15,Paternity Leave\n')
#     fp.write('E108,Linda White,Marketing,2024-08-10,2024-08-20,10,20,10,Compassionate Leave\n')
#     fp.write('E109,David Black,Finance,2024-09-15,2024-09-25,10,20,10,Study Leave\n')

# print(f"Employee leave data has been written to '{file_path}'.")

# # Step 2: Reading the employee leave data into a pandas DataFrame
# df = pd.read_csv(file_path)

# # Convert dates to datetime
# df['Start_Date'] = pd.to_datetime(df['Start_Date'])
# df['End_Date'] = pd.to_datetime(df['End_Date'])

# # Calculate leave duration
# df['Leave_Duration'] = (df['End_Date'] - df['Start_Date']).dt.days + 1

# # Step 3: Performing initial and additional analyses
# df['Leave_Utilization_%'] = (df['Leave_Taken'] / df['Total_Annual_Leave']) * 100

# # Leave duration categories
# def categorize_duration(days):
#     if days <= 5:
#         return 'Short'
#     elif 6 <= days <= 10:
#         return 'Medium'
#     else:
#         return 'Long'

# df['Duration_Category'] = df['Leave_Duration'].apply(categorize_duration)

# # Top 3 employees who have taken the most leave
# top_3_employees = df.nlargest(3, 'Leave_Taken')[['Employee_Name', 'Leave_Taken', 'Reason']]

# # Month with the most leave requests
# df['Month'] = df['Start_Date'].dt.month
# most_leave_month = df['Month'].value_counts().idxmax()

# # Employees eligible for leave (leave remaining > threshold)
# eligible_for_leave = df[df['Leave_Remaining'] > 5][['Employee_Name', 'Leave_Remaining']]

# # Step 4: Exporting results to a file
# output_path = 'extended_leave_analysis_results.txt'
# with open(output_path, 'w') as out:
#     # Summary of analyses
#     out.write("### Extended Leave Analysis ###\n\n")
    
#     # Top 3 employees
#     out.write("Top 3 Employees Who Have Taken the Most Leave:\n")
#     out.write(top_3_employees.to_string(index=False) + "\n\n")
    
#     # Leave utilization
#     out.write("Leave Utilization Percentage for Each Employee:\n")
#     utilization = df[['Employee_Name', 'Leave_Utilization_%']]
#     out.write(utilization.to_string(index=False) + "\n\n")
    
#     # Month with most leave requests
#     out.write(f"Month with the Most Leave Requests: {most_leave_month} (Month Number)\n\n")
    
#     # Leave duration categories
#     out.write("Leave Duration Categorization:\n")
#     categorized_durations = df[['Employee_Name', 'Leave_Duration', 'Duration_Category']]
#     out.write(categorized_durations.to_string(index=False) + "\n\n")
    
#     # Employees eligible for leave
#     out.write("Employees Eligible for Leave (Based on Leave Remaining):\n")
#     out.write(eligible_for_leave.to_string(index=False) + "\n\n")

# print(f"Extended leave analysis completed. Results saved to '{output_path}'.")












# import pandas as pd

# # Step 1: Write employee leave data with additional leave types to a file
# file_path = 'employee_leave_data.txt'
# with open(file_path, 'w') as fp:
#     fp.write('Employee_ID,Employee_Name,Start_Date,End_Date,Leave_Taken,Total_Annual_Leave,Leave_Remaining,Reason\n')
#     fp.write('E101,John Doe,2024-01-05,2024-01-10,5,20,15,Annual Leave\n')
#     fp.write('E102,Jane Smith,2024-02-01,2024-02-03,2,20,18,Personal Leave\n')
#     fp.write('E103,Mike Johnson,2024-03-15,2024-03-20,5,20,15,Sick Leave\n')
#     fp.write('E104,Emily Davis,2024-04-05,2024-04-07,2,20,18,Casual Leave\n')
#     fp.write('E105,James Wilson,2024-05-10,2024-05-14,4,20,16,Annual Leave\n')
#     fp.write('E106,Sarah Brown,2024-06-01,2024-06-30,30,20,-10,Maternity Leave\n')
#     fp.write('E107,Robert Green,2024-07-01,2024-07-05,5,20,15,Paternity Leave\n')
#     fp.write('E108,Linda White,2024-08-10,2024-08-20,10,20,10,Compassionate Leave\n')
#     fp.write('E109,David Black,2024-09-15,2024-09-25,10,20,10,Study Leave\n')

# print(f"Employee leave data has been written to '{file_path}'.")

# # Step 2: Load the data into a pandas DataFrame
# df = pd.read_csv(file_path)

# # Export analysis results to a text file
# output_path = 'leave_analysis_results.txt'
# with open(output_path, 'w') as out:

#     # Total leave taken by all employees
#     total_leave_taken = df['Leave_Taken'].sum()
#     out.write(f"Total Leave Taken by All Employees: {total_leave_taken} days\n\n")

#     # Leave remaining for each employee
#     df['Leave_Remaining'] = df['Total_Annual_Leave'] - df['Leave_Taken']
#     leave_remaining_by_employee = df[['Employee_Name', 'Leave_Remaining']]
#     out.write("Leave Remaining for Each Employee:\n")
#     out.write(leave_remaining_by_employee.to_string(index=False) + "\n\n")

#     # Employee(s) with the most leave taken
#     max_leave_taken = df[df['Leave_Taken'] == df['Leave_Taken'].max()]
#     out.write("Employee(s) Who Have Taken the Most Leave:\n")
#     out.write(max_leave_taken[['Employee_Name', 'Leave_Taken']].to_string(index=False) + "\n\n")

#     # Leave taken grouped by reason
#     grouped_leave_reason = df.groupby('Reason')['Leave_Taken'].sum().reset_index()
#     out.write("Leave Taken Grouped by Reason:\n")
#     out.write(grouped_leave_reason.to_string(index=False) + "\n\n")

#     # Employees with negative leave balances
#     negative_balance = df[df['Leave_Remaining'] < 0]
#     out.write("Employees with Negative Leave Balances:\n")
#     out.write(negative_balance[['Employee_Name', 'Leave_Taken', 'Leave_Remaining', 'Reason']].to_string(index=False) + "\n\n")

#     # Leave trends by month
#     df['Start_Date'] = pd.to_datetime(df['Start_Date'])
#     df['Month'] = df['Start_Date'].dt.month
#     leave_trends = df.groupby('Month').size().reset_index(name='Leave_Count')
#     out.write("Leave Trends by Month:\n")
#     out.write(leave_trends.to_string(index=False) + "\n\n")

#     # Employees taking leave each month
#     monthly_leave_count = df.groupby('Month')['Employee_Name'].nunique().reset_index()
#     monthly_leave_count.columns = ['Month', 'Number_of_Employees']
#     out.write("Number of Employees Taking Leave Each Month:\n")
#     out.write(monthly_leave_count.to_string(index=False) + "\n\n")

#     # Employees who have used all their leave
#     all_leave_used = df[df['Leave_Remaining'] == 0]
#     out.write("Employees Who Have Used All Their Leave:\n")
#     out.write(all_leave_used[['Employee_Name', 'Leave_Taken', 'Leave_Remaining', 'Reason']].to_string(index=False) + "\n\n")

#     # Total leave taken by each employee
#     leave_taken_by_employee = df.groupby('Employee_Name')['Leave_Taken'].sum().reset_index()
#     leave_taken_by_employee.columns = ['Employee_Name', 'Total_Leave_Taken']
#     out.write("Total Leave Taken by Each Employee:\n")
#     out.write(leave_taken_by_employee.to_string(index=False) + "\n\n")

#     # Employees who haven't taken any leave
#     no_leave_taken = df[df['Leave_Taken'] == 0]
#     out.write("Employees Who Have Not Taken Any Leave:\n")
#     out.write(no_leave_taken[['Employee_Name', 'Leave_Remaining']].to_string(index=False) + "\n")

# # Define a threshold for excessive leave
# leave_threshold = 15

# # Employees exceeding the leave threshold
# exceeding_threshold = df[df['Leave_Taken'] > leave_threshold]
# print("Employees Exceeding the Leave Threshold:")
# print(exceeding_threshold[['Employee_Name', 'Leave_Taken', 'Reason']])

# # Average leave duration
# avg_leave_duration = df['Leave_Duration'].mean()
# print(f"\nAverage Leave Duration: {avg_leave_duration:.2f} days")

# # Distribution of leave by reason with average leave duration
# reason_leave_distribution = df.groupby('Reason').agg(
#     Total_Leave=('Leave_Taken', 'sum'),
#     Average_Duration=('Leave_Duration', 'mean'),
#     Number_of_Employees=('Reason', 'size')
# ).reset_index()
# print("\nDistribution of Leave by Reason:")
# print(reason_leave_distribution)

# # Monthly trends in leave duration
# monthly_leave_duration = df.groupby('Month')['Leave_Duration'].sum().reset_index()
# print("\nMonthly Trends in Leave Duration:")
# print(monthly_leave_duration)

# # Employees with leave balances falling within certain ranges
# balance_ranges = {
#     'Above 10 days': (df['Leave_Remaining'] > 10).sum(),
#     'Between 0 and 10 days': ((df['Leave_Remaining'] > 0) & (df['Leave_Remaining'] <= 10)).sum(),
#     'Negative Balance': (df['Leave_Remaining'] < 0).sum(),
# }
# print("\nLeave Balance Ranges Summary:")
# for category, count in balance_ranges.items():
#     print(f"{category}: {count} employees")

# # Analysis of leave duration outliers using Z-score
# from scipy.stats import zscore

# df['Leave_Duration_Z'] = zscore(df['Leave_Duration'])
# outliers = df[abs(df['Leave_Duration_Z']) > 2]
# print("\nLeave Duration Outliers (Z-Score Method):")
# print(outliers[['Employee_Name', 'Leave_Duration', 'Leave_Duration_Z']])


# print(f"Analysis results have been exported to '{output_path}'.")










# import pandas as pd

# # Step 1: Creating and writing employee leave data (with additional leave types) to a file
# with open('employee_leave_data.txt', 'w') as fp:
#     fp.write('Employee_ID,Employee_Name,Start_Date,End_Date,Leave_Taken,Total_Annual_Leave,Leave_Remaining,Reason\n')
#     fp.write('E101,John Doe,2024-01-05,2024-01-10,5,20,15,Annual Leave\n')
#     fp.write('E102,Jane Smith,2024-02-01,2024-02-03,2,20,18,Personal Leave\n')
#     fp.write('E103,Mike Johnson,2024-03-15,2024-03-20,5,20,15,Sick Leave\n')
#     fp.write('E104,Emily Davis,2024-04-05,2024-04-07,2,20,18,Casual Leave\n')
#     fp.write('E105,James Wilson,2024-05-10,2024-05-14,4,20,16,Annual Leave\n')
#     fp.write('E106,Sarah Brown,2024-06-01,2024-06-30,30,20,-10,Maternity Leave\n')
#     fp.write('E107,Robert Green,2024-07-01,2024-07-05,5,20,15,Paternity Leave\n')
#     fp.write('E108,Linda White,2024-08-10,2024-08-20,10,20,10,Compassionate Leave\n')
#     fp.write('E109,David Black,2024-09-15,2024-09-25,10,20,10,Study Leave\n')

# print("Employee leave data with additional leave types has been written to 'employee_leave_data.txt'")



# # Step 

# #Reading the employee leave data into a pandas DataFrame

# import pandas as pd

# # Reading the data
# df = pd.read_csv('employee_leave_data.txt')

# # Display the data
# print("Employee Leave Data with Additional Leave Types:")
# print(df)



# #Summing Total Leave Taken by All Employees

# # Total leave taken by all employees
# total_leave_taken = df['Leave_Taken'].sum()
# print(f"\nTotal Leave Taken by All Employees: {total_leave_taken} days")




# # Calculating leave remaining for each employee
# df['Leave_Remaining'] = df['Total_Annual_Leave'] - df['Leave_Taken']

# # Creating a DataFrame to track remaining leave

# leave_remaining_by_employee = df[['Employee_Name', 'Leave_Remaining']]

# # Display the result
# print("Leave Remaining for Each Employee:")
# print(leave_remaining_by_employee)



# # Finding the employee who has taken the most leave

# max_leave_taken = df[df['Leave_Taken'] == df['Leave_Taken'].max()]

# print("\nEmployee(s) Who Have Taken the Most Leave:")
# print(max_leave_taken[['Employee_Name', 'Leave_Taken']])



# # Grouping by the type of leave (e.g., Maternity Leave, Sick Leave, etc.)

# grouped_leave_reason = df.groupby('Reason')['Leave_Taken'].sum().reset_index()

# print("\nLeave Taken Grouped by Reason:")
# print(grouped_leave_reason)




# # Identifying employees with negative leave balances

# negative_balance = df[df['Leave_Remaining'] < 0]

# print("\nEmployees with Negative Leave Balances:")

# print(negative_balance[['Employee_Name', 'Leave_Taken', 'Leave_Remaining', 'Reason']])




# # Calculate leave distribution by reason with employee names
# leave_distribution_by_reason = df.groupby('Reason').agg(

#     Count=('Reason', 'size'), 

#     Employees=('Employee_Name', ', '.join) 

# ).reset_index()




# #Analyze Leave Trends Over Time (Monthly): which months experience the highest leave requests

# # Converting 'Start_Date' to datetime format

# df['Start_Date'] = pd.to_datetime(df['Start_Date'])

# # Extracting the month and analyzing trends by month

# df['Month'] = df['Start_Date'].dt.month

# leave_trends = df.groupby('Month').size().reset_index(name='Leave_Count')

# print("\nLeave Trends by Month:")
# print(leave_trends)




# #To determine the number of employees taking leave each month

# # Convert 'Start_Date' to datetime format
# df['Start_Date'] = pd.to_datetime(df['Start_Date'])

# # Extract month from 'Start_Date' and add it as a new column
# df['Month'] = df['Start_Date'].dt.month

# # Group by the 'Month' column and count unique 'Employee_Name's in each month
# monthly_leave_count = df.groupby('Month')['Employee_Name'].nunique().reset_index()

# # Rename columns for clarity
# monthly_leave_count.columns = ['Month', 'Number_of_Employees']

# # Display results
# print("Number of employees taking leave each month:")
# print(monthly_leave_count)



# # Detecting employees who have used up all their annual leave

# all_leave_used = df[df['Leave_Remaining'] == 0]

# print("\nEmployees Who Have Used All Their Leave:")
# print(all_leave_used[['Employee_Name', 'Leave_Taken', 'Leave_Remaining', 'Reason']])




# # Calculating total leave taken by each employee
# leave_taken_by_employee = df.groupby('Employee_Name')['Leave_Taken'].sum().reset_index()

# # Renaming columns for clarity
# leave_taken_by_employee.columns = ['Employee_Name', 'Total_Leave_Taken']


# # Display the result
# print(leave_taken_by_employee)




# #Detect employees who haven't taken any leave

# no_leave_taken = df[df['Leave_Taken'] == 0]

# print("Employees Who Have Not Taken Any Leave:")
# print(no_leave_taken[['Employee_Name', 'Leave_Remaining']])




















# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
# import os

# # Read the Excel file into a DataFrame
# try:
#     df = pd.read_excel('monthly-analysis.xlsx')
# except FileNotFoundError:
#     print("Error: The file 'monthly-analysis.xlsx' was not found.")
#     exit()

# # Check for required columns
# required_columns = ["OBSERVATION RATING", "NUMBER OF TIME", "USERS", "USER REASON", "PHONE", "REQUEST MEDIUM"]
# missing_columns = [col for col in required_columns if col not in df.columns]
# if missing_columns:
#     print(f"Error: Missing columns {missing_columns} in the data.")
#     exit()

# # Ensure phone numbers are treated as strings and preserve leading zeros
# if 'PHONE' in df.columns:
#     df['PHONE'] = df['PHONE'].astype(str).str.zfill(11)  # Adjust to the required length
# else:
#     print("Warning: 'PHONE' column not found in the dataset.")


# # Total Requests
# total_requests = len(df)
# print(f"The Total self-exclusion requests for the month: {total_requests}\n")

# # Standardize the 'OBSERVATION RATING' column to lowercase
# df["OBSERVATION RATING"] = df["OBSERVATION RATING"].str.lower()

# # Observation Rating Counts
# rating_counts = df["OBSERVATION RATING"].value_counts()

# # Low count sorted by "NUMBER OF TIME"
# Lowcount_sorted = df[df["OBSERVATION RATING"] == 'low'].sort_values(by="NUMBER OF TIME", ascending=False)
# low_count_filtered = Lowcount_sorted[["USERS", "USER REASON", "NUMBER OF TIME", "OBSERVATION RATING", "PHONE"]]
# print("Low Observation Ratings (Sorted by Number of Times):")
# print(low_count_filtered)
# print()

# # Medium count sorted by "NUMBER OF TIME"
# medium_count = df[df["OBSERVATION RATING"] == 'medium'].sort_values(by="NUMBER OF TIME", ascending=False)
# medium_count_filtered = medium_count[["USERS", "USER REASON", "NUMBER OF TIME", "OBSERVATION RATING", "PHONE"]]
# print("Medium Observation Ratings (Sorted by Number of Times):")
# print(medium_count_filtered)
# print()

# # Group by "REQUEST MEDIUM"
# grouped_simple = df.groupby("REQUEST MEDIUM").size().reset_index(name="Count")
# grouped_simple.columns = ["REQUEST MEDIUM", "Occurrence"]
# print("Request Medium Grouping:")
# print(grouped_simple)
# print()

# grouped_medium = grouped_simple.sort_values(by="Occurrence", ascending=False)

# # Group by "USER REASON"
# grouped_by_reason = (
#     df.groupby("USER REASON")
#     .size()
#     .reset_index(name="Count")
#     .sort_values(by="Count", ascending=False)
#     .rename(columns={"USER REASON": "User Reason"})
# )

# # Save results to Excel and include graphs
# output_file = "monthly-analysis-output.xlsx"
# with pd.ExcelWriter(output_file, engine='xlsxwriter') as writer:
#     # Save data
#     df.to_excel(writer, sheet_name="Main Data", index=False)
#     pd.DataFrame({"Total Requests": [total_requests]}).to_excel(writer, sheet_name="Summary", index=False)
#     rating_counts.to_frame().to_excel(writer, sheet_name="Rating Counts")
#     grouped_medium.to_excel(writer, sheet_name="Request Medium", index=False)
#     grouped_by_reason.to_excel(writer, sheet_name="User Reasons", index=False)
#     low_count_filtered.to_excel(writer, sheet_name="Low Ratings", index=False)
#     medium_count_filtered.to_excel(writer, sheet_name="Medium Ratings", index=False)

#     # Access the workbook and worksheet
#     workbook = writer.book
#     worksheet = workbook.add_worksheet("Graphs")
#     writer.sheets['Graphs'] = worksheet

#     # Generate and save graphs as images
#     graph_files = []

#     # Observation Rating Bar Chart
#     plt.figure(figsize=(8, 6))
#     sns.barplot(x=rating_counts.index, y=rating_counts.values, palette="viridis")
#     plt.title("Observation Rating Counts")
#     plt.xlabel("Observation Rating")
#     plt.ylabel("Count")
#     bar_chart_file = "observation_rating_counts.png"
#     plt.savefig(bar_chart_file)
#     graph_files.append(bar_chart_file)
#     plt.close()

#     # Request Medium Pie Chart
#     plt.figure(figsize=(8, 8))
#     plt.pie(
#         grouped_medium["Occurrence"],
#         labels=grouped_medium["REQUEST MEDIUM"],
#         autopct="%1.1f%%",
#         startangle=140,
#         colors=sns.color_palette("pastel"),
#     )
#     plt.title("Distribution of Request Medium")
#     pie_chart_file = "request_medium_distribution.png"
#     plt.savefig(pie_chart_file)
#     graph_files.append(pie_chart_file)
#     plt.close()

#     # User Reasons Bar Chart
#     plt.figure(figsize=(10, 6))
#     sns.barplot(x=grouped_by_reason["Count"], y=grouped_by_reason["User Reason"], palette="coolwarm")
#     plt.title("Top User Reasons")
#     plt.xlabel("Count")
#     plt.ylabel("User Reason")
#     user_reason_chart_file = "top_user_reasons.png"
#     plt.savefig(user_reason_chart_file)
#     graph_files.append(user_reason_chart_file)
#     plt.close()

#     # Line Graph for Requests Over Time (Example: Using 'DATE' Column)
#     if "DATE" in df.columns:
#         df["DATE"] = pd.to_datetime(df["DATE"])  # Ensure DATE is in datetime format
#         requests_over_time = df.groupby(df["DATE"].dt.date).size()
#         plt.figure(figsize=(10, 6))
#         requests_over_time.plot(kind="line", marker="o", color="blue")
#         plt.title("Requests Over Time")
#         plt.xlabel("Date")
#         plt.ylabel("Request Count")
#         plt.grid(True)
#         line_chart_file = "requests_over_time.png"
#         plt.savefig(line_chart_file)
#         graph_files.append(line_chart_file)
#         plt.close()
#     else:
#         print("No 'DATE' column found. Skipping line graph.")

#     # Insert images into the "Graphs" worksheet
#     for i, graph_file in enumerate(graph_files):
#         worksheet.insert_image(f"A{i * 20 + 1}", graph_file)

# # Clean up temporary graph files
# for graph_file in graph_files:
#     os.remove(graph_file)

# print(f"Data and graphs have been saved to '{output_file}'")














# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt

# # Read the Excel file into a DataFrame
# try:
#     df = pd.read_excel('monthly-analysis.xlsx')
# except FileNotFoundError:
#     print("Error: The file 'monthly-analysis.xlsx' was not found.")
#     exit()

# # Total Requests
# total_requests = len(df)
# print(f"The Total self-exclusion requests for the month: {total_requests}\n")

# # Standardize the 'OBSERVATION RATING' column to lowercase
# df["OBSERVATION RATING"] = df["OBSERVATION RATING"].str.lower()

# # Observation Rating Counts
# rating_counts = df["OBSERVATION RATING"].value_counts()


#  #Low count sorted by "NUMBER OF TIME"

# Lowcount_sorted = df[df["OBSERVATION RATING"].str.lower() == 'low'].sort_values(by="NUMBER OF TIME", ascending=False)

# low_count_sorted = df[["USERS", "USER REASON", "NUMBER OF TIME", "OBSERVATION RATING", "PHONE"]]

# print(Lowcount_sorted)
# print()


# # Medium count sorted by "NUMBER OF TIME"

# mediumcount = df[df["OBSERVATION RATING"] == 'medium'].sort_values(by="NUMBER OF TIME", ascending=False)

# mediumcount = df[["USERS", "USER REASON", "NUMBER OF TIME", "OBSERVATION RATING", "PHONE"]]

# print(mediumcount)
# print()



# # Group by "REQUEST MEDIUM" only
# grouped_simple = df.groupby("REQUEST MEDIUM").size().reset_index(name="Count")
# grouped_simple.columns = ["REQUEST MEDIUM", "Occurrence"]
# print(grouped_simple)
# print()



# # Group by "REQUEST MEDIUM"
# grouped_medium = df.groupby("REQUEST MEDIUM").size().reset_index(name="Count").sort_values(by="Count", ascending=False)
# grouped_medium.columns = ["REQUEST MEDIUM", "Occurrence"]

# # Group by "USER REASON"
# grouped_by_reason = (
#     df.groupby("USER REASON")
#     .size()
#     .reset_index(name="Count")
#     .sort_values(by="Count", ascending=False)
#     .rename(columns={"USER REASON": "User Reason"})
# )

# # Save results to Excel and include graphs
# output_file = "monthly-analysis-output.xlsx"
# with pd.ExcelWriter(output_file, engine='xlsxwriter') as writer:
#     # Save data
#     df.to_excel(writer, sheet_name="Main Data", index=False)
#     pd.DataFrame({"Total Requests": [total_requests]}).to_excel(writer, sheet_name="Summary", index=False)
#     rating_counts.to_frame().to_excel(writer, sheet_name="Rating Counts")
#     grouped_medium.to_excel(writer, sheet_name="Request Medium", index=False)
#     grouped_by_reason.to_excel(writer, sheet_name="User Reasons", index=False)

#     # Access the workbook and worksheet
#     workbook = writer.book
#     worksheet = workbook.add_worksheet("Graphs")
#     writer.sheets['Graphs'] = worksheet

#     # Generate and save graphs as images
#     graph_files = []

#     # Observation Rating Bar Chart
#     plt.figure(figsize=(8, 6))
#     sns.barplot(x=rating_counts.index, y=rating_counts.values, palette="viridis")
#     plt.title("Observation Rating Counts")
#     plt.xlabel("Observation Rating")
#     plt.ylabel("Count")
#     bar_chart_file = "observation_rating_counts.png"
#     plt.savefig(bar_chart_file)
#     graph_files.append(bar_chart_file)
#     plt.close()

#     # Request Medium Pie Chart
#     plt.figure(figsize=(8, 8))
#     plt.pie(
#         grouped_medium["Occurrence"],
#         labels=grouped_medium["REQUEST MEDIUM"],
#         autopct="%1.1f%%",
#         startangle=140,
#         colors=sns.color_palette("pastel"),
#     )
#     plt.title("Distribution of Request Medium")
#     pie_chart_file = "request_medium_distribution.png"
#     plt.savefig(pie_chart_file)
#     graph_files.append(pie_chart_file)
#     plt.close()

#     # User Reasons Bar Chart
#     plt.figure(figsize=(10, 6))
#     sns.barplot(x=grouped_by_reason["Count"], y=grouped_by_reason["User Reason"], palette="coolwarm")
#     plt.title("Top User Reasons")
#     plt.xlabel("Count")
#     plt.ylabel("User Reason")
#     user_reason_chart_file = "top_user_reasons.png"
#     plt.savefig(user_reason_chart_file)
#     graph_files.append(user_reason_chart_file)
#     plt.close()

#     # Line Graph for Requests Over Time (Example: Using 'DATE' Column)
#     if "DATE" in df.columns:
#         df["DATE"] = pd.to_datetime(df["DATE"])  # Ensure DATE is in datetime format
#         requests_over_time = df.groupby(df["DATE"].dt.date).size()
#         plt.figure(figsize=(10, 6))
#         requests_over_time.plot(kind="line", marker="o", color="blue")
#         plt.title("Requests Over Time")
#         plt.xlabel("Date")
#         plt.ylabel("Request Count")
#         plt.grid(True)
#         line_chart_file = "requests_over_time.png"
#         plt.savefig(line_chart_file)
#         graph_files.append(line_chart_file)
#         plt.close()
#     else:
#         print("No 'DATE' column found. Skipping line graph.")

#     # Insert images into the "Graphs" worksheet
#     for i, graph_file in enumerate(graph_files):
#         worksheet.insert_image(f"A{i * 20 + 1}", graph_file)

# print(f"Data and graphs have been saved to '{output_file}'")













# import pandas as pd
# from xlsxwriter import Workbook
# import matplotlib.pyplot as plt


# # Read the Excel file into a DataFrame
# df = pd.read_excel('monthly-analysis.xlsx')


# # Read the Excel file into a DataFrame
# df = pd.read_excel('monthly-analysis.xlsx')


# # 1. Total Requests
# total_requests = len(df)
# print(f" The Total self-exclusion request for the Month is {total_requests}")
# print()

# # Convert all values in the 'OBSERVATION RATING' column to lowercase
# df["OBSERVATION RATING"] = df["OBSERVATION RATING"].str.lower()

# # Count the occurrences of each rating
# rating_counts = df["OBSERVATION RATING"].value_counts()

# # Print the counts
# print(rating_counts)
# print()

# # Medium count sorted by "NUMBER OF TIME"
# mediumcount = df[df["OBSERVATION RATING"] == 'medium'].sort_values(by="NUMBER OF TIME", ascending=False)
# print(mediumcount)
# print()

# # Low count sorted by "NUMBER OF TIME"
# Lowcount_sorted = df[df["OBSERVATION RATING"].str.lower() == 'low'].sort_values(by="NUMBER OF TIME", ascending=False)
# print(Lowcount_sorted)
# print()




# # Group by "REQUEST MEDIUM" only
# grouped_simple = df.groupby("REQUEST MEDIUM").size().reset_index(name="Count")
# grouped_simple.columns = ["REQUEST MEDIUM", "Occurrence"]
# print(grouped_simple)
# print()


# # Group by "REQUEST MEDIUM" only
# grouped_simple = df.groupby("REQUEST MEDIUM").size().reset_index(name="Count")
# grouped_simple.columns = ["REQUEST MEDIUM", "Occurrence"]
# print(grouped_simple)
# print()

# # Group by "USER REASON"
# grouped_by_reason = df.groupby("USER REASON").size().reset_index(name="Count")

# # Sort the grouped result by "Count" in descending order
# grouped_by_reason = grouped_by_reason.sort_values(by="Count", ascending=False)

# # Rename the columns for clarity
# grouped_by_reason.columns = ["USER REASON", "COUNT"]

# # Print the grouped and sorted DataFrame
# print(grouped_by_reason)


# # Save the final DataFrame to Excel with multiple sheets
# output_file = "monthly.xlsx"

# with pd.ExcelWriter(output_file, engine='xlsxwriter') as writer:
#     # Save the main DataFrame to a sheet
#     df.to_excel(writer, sheet_name="Main Data", index=False)

#     # Save additional outputs to separate sheets
#     pd.DataFrame({"Total Requests": [total_requests]}).to_excel(writer, sheet_name="Summary", index=False)

#     # Save the counts and filtered data to separate sheets
#     rating_counts.to_frame().to_excel(writer, sheet_name="Rating Counts")
#     mediumcount.to_excel(writer, sheet_name="Medium Count")
#     Lowcount_sorted.to_excel(writer, sheet_name="Low Count Sorted")
#     grouped_simple.to_excel(writer, sheet_name="Grouped by Medium")
#     grouped_by_reason.to_excel(writer, sheet_name="Grouped by Reason")

# print(f"Data has been saved to {output_file}")






# import pandas as pd

# # Read the Excel file into a DataFrame
# df = pd.read_excel('monthly-analysis.xlsx')


# df['MOBILE NUMBER'] = df['MOBILE NUMBER'].astype(str)

# # You can also use this to remove any leading/trailing spaces
# df['MOBILE NUMBER'] = df['MOBILE NUMBER'].str.strip()

# # 1. Total Requests
# total_requests = len(df)
# print(f" The Total self-exclusion request for the Month is {total_requests}")
# print()

# # Convert all values in the 'OBSERVATION RATING' column to lowercase
# df["OBSERVATION RATING"] = df["OBSERVATION RATING"].str.lower()

# # Count the occurrences of each rating
# rating_counts = df["OBSERVATION RATING"].value_counts()

# # Print the counts
# print(rating_counts)
# print()

# # Medium count sorted by "NUMBER OF TIME"
# mediumcount = df[df["OBSERVATION RATING"] == 'medium'].sort_values(by="NUMBER OF TIME", ascending=False)
# print(mediumcount)
# print()

# # Low count sorted by "NUMBER OF TIME"
# Lowcount_sorted = df[df["OBSERVATION RATING"].str.lower() == 'low'].sort_values(by="NUMBER OF TIME")
# print(Lowcount_sorted)
# print()

# # Group by "REQUEST MEDIUM" only
# grouped_simple = df.groupby("REQUEST MEDIUM").size().reset_index(name="Count")
# grouped_simple.columns = ["REQUEST MEDIUM", "Occurrence"]
# print(grouped_simple)
# print()

# # Group by "USER REASON"
# grouped_by_reason = df.groupby("USER REASON").size().reset_index(name="Count")
# grouped_by_reason.columns = ["USER REASON", "COUNT"]
# print(grouped_by_reason)
# print()

# # Save the final DataFrame to CSV
# output_file = "monthlyanalysis.csv"
# df.to_csv(output_file, index=False)

# # Save the counts and filtered data to separate CSV files
# rating_counts.to_csv('rating_counts.csv', header=True)
# mediumcount.to_csv('medium_count.csv', index=False)
# Lowcount_sorted.to_csv('low_count_sorted.csv', index=False)
# grouped_simple.to_csv('grouped_simple.csv', index=False)
# grouped_by_reason.to_csv('grouped_by_reason.csv', index=False)

# print(f"Data has been saved to {output_file}")
# print("Additional outputs saved to separate CSV files.")
















# import pandas as pd

# # Sample sales data
# data = {
#     'Customer': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
#     'Product': ['Rice', 'Beans', 'Rice', 'Garri', 'Rice'],
#     'Quantity_kg': [25, 5, 10, 8, 15],  # Quantity sold in kilograms
#     'Payment_Method': ['Cash', 'Transfer', 'Cash', 'Transfer', 'Cash'],
#     'Date': ['2024-11-01', '2024-12-02', '2024-12-03', '2024-12-04', '2024-12-05'], # Sale dates
#     'Amount_Paid': [12500, 2500, 5000, 4000, 7500]  # Amount paid in currency
# }

# # Create a DataFrame
# df = pd.DataFrame(data)

# # Define initial inventory for each product
# initial_inventory = {
#     'Rice': 100,
#     'Beans': 20,
#     'Garri': 15
# }

# # Convert 'Date' column to datetime
# df['Date'] = pd.to_datetime(df['Date'])

# # Weekly Revenue
# df['Week'] = df['Date'].dt.isocalendar().week

# # Create a year column to handle weeks spanning different years
# df['Year'] = df['Date'].dt.year

# # Define a function to generate meaningful week labels
# def week_label(year, week):
#     start_date = pd.Timestamp(f'{year}-01-01') + pd.offsets.Week(weekday=0) * (week - 1)
#     end_date = start_date + pd.Timedelta(days=6)
#     return f"{start_date.strftime('%b %d')} - {end_date.strftime('%b %d')}"

# # Add meaningful week labels
# df['Week_Label'] = df.apply(lambda x: week_label(x['Year'], x['Week']), axis=1)

# # Group by Week_Label instead of Week
# weekly_revenue = df.groupby('Week_Label')['Amount_Paid'].sum()

# # Monthly Revenue
# df['Month'] = df['Date'].dt.month
# monthly_revenue = df.groupby('Month')['Amount_Paid'].sum()

# # Display the final DataFrame with all details
# print("Sales Data with Inventory and Remaining Stock:")
# print(df)

# # Display Weekly Revenue with meaningful labels
# print("\nWeekly Revenue:")
# print(weekly_revenue)

# # Display Monthly Revenue
# print("\nMonthly Revenue:")
# print(monthly_revenue)







# import pandas as pd

# # Sample sales data
# data = {
#     'Customer': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
#     'Product': ['Rice', 'Beans', 'Rice', 'Garri', 'Rice'],
#     'Quantity_kg': [25, 5, 10, 8, 15],  # Quantity sold in kilograms
#     'Payment_Method': ['Cash', 'Transfer', 'Cash', 'Transfer', 'Cash'],
#     'Date': ['2024-11-01', '2024-12-02', '2024-12-03', '2024-12-04', '2024-12-05'], # Sale dates
#     'Amount_Paid': [12500, 2500, 5000, 4000, 7500]  # Amount paid in currency
# }

# # Create a DataFrame
# df = pd.DataFrame(data)

# # Define initial inventory for each product
# initial_inventory = {
#     'Rice': 100,
#     'Beans': 20,
#     'Garri': 15
# }

# # Convert 'Date' column to datetime
# df['Date'] = pd.to_datetime(df['Date'])

# # Weekly Revenue
# df['Week'] = df['Date'].dt.isocalendar().week
# weekly_revenue = df.groupby('Week')['Amount_Paid'].sum()

# # Monthly Revenue
# df['Month'] = df['Date'].dt.month
# monthly_revenue = df.groupby('Month')['Amount_Paid'].sum()

# # Product Performance
# product_performance = df.groupby('Product')['Amount_Paid'].sum().sort_values(ascending=False)

# # Map initial inventory to the products in the DataFrame
# df['Initial_Inventory'] = df['Product'].map(initial_inventory)

# # Group by product to calculate the total quantity sold
# total_sold = df.groupby('Product')['Quantity_kg'].sum()

# # Initialize an empty dictionary to store remaining stock
# remaining_stock = {}

# # Calculate remaining stock for each product and update the dictionary
# for product in total_sold.index:
#     remaining_stock[product] = initial_inventory[product] - total_sold[product]

# # Convert the remaining stock dictionary to a DataFrame for summary
# remaining_stock_summary = pd.DataFrame(list(remaining_stock.items()), columns=['Product', 'Remaining_Stock'])

# # Map remaining stock back to the original DataFrame
# df['Remaining_Stock'] = df['Product'].map(remaining_stock)

# # Display the final DataFrame with all details
# print("Sales Data with Inventory and Remaining Stock:")
# print(df)

# # Display summarized remaining stock
# print("\nRemaining Stock Summary:")
# print(remaining_stock_summary)

# # Display Weekly Revenue
# print("\nWeekly Revenue:")
# print(weekly_revenue)

# # Display Monthly Revenue
# print("\nMonthly Revenue:")
# print(monthly_revenue)


















# import pandas as pd

# # Sample data
# data = {
#     'day': ['Monday', 'Monday', 'Tuesday', 'Tuesday', 'Wednesday', 'Wednesday'],
#     'department': ['Emergency', 'Cardiology', 'Emergency', 'Orthopedics', 'Cardiology', 'Emergency'],
#     'patients': [30, 10, 25, 20, 15, 40]
# }

# df = pd.DataFrame(data)

# # print(df)
# # print()

# pivot = df.pivot_table(index='day', columns='department', values='patients', aggfunc='sum', fill_value=5)

# print(pivot)

# import matplotlib.pyplot as plt


# pivot.plot(kind='bar')
# plt.title('Patient Count by Department and Day')
# plt.xlabel('Day')
# plt.ylabel('Number of Patients')
# plt.xticks(rotation=45)  # Rotate x-axis labels for better visibility
# plt.tight_layout()  # Adjust the layout to make room for x-axis labels
# plt.show()

# pivot.plot(kind='bar')
# plt.title('Patient Count by Department and Day')
# plt.xlabel('Day')
# plt.ylabel('Number of Patients')
# plt.show()







# import pandas as pd
# import matplotlib.pyplot as plt

# # Sample data with Leave Taken included
# data = {
#     'Employee': ['Alice', 'Bob', 'Charlie', 'David', 'Eve', 'Frank', 'Grace', 'Hannah', 'Ivy', 'Jack'],
#     'Leave Balance': [18, 12, 8, 3, 16, 14, 5, 7, 20, 2],
#     'Leave Taken': [5, 8, 12, 15, 7, 10, 18, 14, 3, 20]  # Example Leave Taken data
# }
# df = pd.DataFrame(data)

# # Calculate Leave Taken from Leave Balance (for demonstration purposes)
# df['Leave Taken'] = df['Leave Balance'] - df['Leave Balance'].min()  # Replace with actual leave taken data if available

# # Sort DataFrame by 'Leave Taken' in ascending order
# df_sorted = df.sort_values(by='Leave Taken')

# # Define colors based on Leave Taken
# def leave_taken_color(taken):
#     if taken > 15:
#         return 'green'
#     elif 10 <= taken <= 15:
#         return 'blue'
#     elif 5 <= taken < 10:
#         return 'orange'
#     else:
#         return 'red'

# df_sorted['Color'] = df_sorted['Leave Taken'].apply(leave_taken_color)

# # Plot the bar chart
# plt.figure(figsize=(12, 7))
# bar_width = 0.35  # Width of the bars
# index = range(len(df_sorted))

# # Create bars for Leave Balance and Leave Taken
# plt.bar(index, df_sorted['Leave Balance'], bar_width, color='lightgrey', label='Leave Balance')
# plt.bar([i + bar_width for i in index], df_sorted['Leave Taken'], bar_width, color=df_sorted['Color'], label='Leave Taken')

# # Adding labels and title
# plt.xlabel('Employee')
# plt.ylabel('Number of Days')
# plt.title('Leave Balance and Leave Taken of Employees (Sorted by Leave Taken)')
# plt.xticks([i + bar_width / 2 for i in index], df_sorted['Employee'], rotation=45)

# # Adding a legend
# legend_patches = [
#     plt.Line2D([0], [0], color='green', lw=4, label='More than 15 Days'),
#     plt.Line2D([0], [0], color='blue', lw=4, label='10-15 Days'),
#     plt.Line2D([0], [0], color='orange', lw=4, label='5-9 Days'),
#     plt.Line2D([0], [0], color='red', lw=4, label='Less than 5 Days')
# ]
# plt.legend(handles=legend_patches)

# # Show plot
# plt.tight_layout()
# plt.show()

# # Display the sorted DataFrame
# print(df_sorted)











# import pandas as pd
# import matplotlib.pyplot as plt

# # Data to be visualized
# data = {
#     'Name': [
#         'Wasiu', 'Marvelous', 'Orire', 'Eniola', 'Eberechukwu', 'Teslimot', 
#         'Oluwaseun', 'Chinazor', 'Celestina', 'Abiakalam', 'Anna', 'Lynda', 'Tinuola', 
#         'Soniran', 'George', 'Chukwudi', 'Martha', 'Amala Okolo', 'Blessing', 'Selimotu', 
#         'Isaac', 'Toyin', 'Jane', 'Ganiyu', 'Jagun', 'Abimbola A', 'Abiola', 'Michael', 
#         'Sarah', 'Momodu', 'Seye', 'Habeeb T', 'Chidera', 'Anyanwu', 'Rita', 'Temitope', 'Oni',
#         'Kofo', 'Gbenga', 'Olaniyan F', 'Olaitan', 'Seyi', 'Kevi', 'Nkiru', 'Lucky', 'Adedoyin',
#         'Etipou', 'Olumide', 'Olayinka', 'Regina', 'Alexander', 'Idris', 'Adebayo', 'Benson',
#         'Totosi', 'Gabriel', 'Anyaokei', 'Sandra', 'Samson', 'Arinze', 'Uduak', 'Sulaimon',
#         'Nelson', 'Adetayo', 'Daniel', 'Obiageli', 'Selimat', 'Elizabeth', 'Rachael', 'Saheed',
#         'Sanusi', 'Akintunde', 'Moshood', 'Emy', 'Ogechukwu', 'Adija', 'Ayodele D', 'Fabeyo',
#         'Godwin', 'Lois', 'Olabimpe', 'Funke', 'Akan', 'Vivian', 'Agozue', 'Helen', 'Owolabi',
#         'Kellywealth', 'Isioma', 'Titilayo', 'Felicia', 'Ella', 'Adesewa', 'Afolashade-O', 'Justin',
#         'Happiness', 'Sophia', 'Lilian', 'Fatima', 'Queen', 'Ifeoma', 'Ojuroye', 'Busayo'
#     ],
#     'Days Requested': [
#         11, 10, 10, 22, 20, 10, 12, 7, 12, 17, 10, 12, 15, 12, 10, 10, 12, 12, 10, 17, 
#         25, 13, 16, 12, 10, 15, 15, 10, 12, 22, 12, 11, 10, 15, 17, 22, 10, 20, 10, 10, 12,
#         17, 17, 12, 10, 10, 10, 22, 22, 10, 12, 20, 10, 11, 10, 10, 14, 10, 22, 15, 12, 10,
#         10, 10, 12, 17, 10, 12, 11, 10, 10, 10, 11, 10, 15, 15, 10, 10, 12, 12, 10, 10, 10,
#         20, 10, 5, 5, 15, 10, 10, 10, 10, 12, 10, 10, 15, 13, 12, 12, 22, 5, 12, 5
#     ]
# }


# # Create a DataFrame
# df = pd.DataFrame(data)

# # Sort the DataFrame by 'Days Requested' in ascending order
# df_sorted = df.sort_values(by='Days Requested')

# # Plot using a horizontal bar chart for better readability of names
# plt.figure(figsize=(20, 16))  # Increase figure size for better readability
# plt.barh(df_sorted['Name'], df_sorted['Days Requested'], color='skyblue', height=0.5)

# # Improve readability of names
# plt.xlabel('Days Requested', fontsize=16)
# plt.ylabel('Name', fontsize=16)
# plt.title('Days Requested by Each Employee', fontsize=18)
# plt.xticks(fontsize=12)
# plt.yticks(fontsize=10)
# plt.subplots_adjust(left=0.2, right=0.8, top=0.9, bottom=0.05)  # Adjust subplot parameters for more space

# # Show grid lines for better readability
# plt.grid(axis='x', linestyle='--', alpha=0.7)

# # Add name labels to bars for better identification
# for index, value in enumerate(df_sorted['Days Requested']):
#     plt.text(value, index, str(value), va='center', ha='left', fontsize=10)

#     plt.show()



# import pandas as pd
# import matplotlib.pyplot as plt

# # Data to be visualized
# data = {
#     'Name': [
#         'Wasiu', 'Marvelous', 'Orire', 'Eniola', 'Eberechukwu', 'Teslimot', 
#         'Oluwaseun', 'Chinazor', 'Celestina', 'Abiakalam', 'Anna', 'Lynda', 'Tinuola', 
#         'Soniran', 'George', 'Chukwudi', 'Martha', 'Amala Okolo', 'Blessing', 'Selimotu', 
#         'Isaac', 'Toyin', 'Jane', 'Ganiyu', 'Jagun', 'Abimbola A', 'Abiola', 'Michael', 
#         'Sarah', 'Momodu', 'Seye', 'Habeeb T', 'Chidera', 'Anyanwu', 'Rita', 'Temitope', 'Oni',
#         'Kofo', 'Gbenga', 'Olaniyan F', 'Olaitan', 'Seyi', 'Kevi', 'Nkiru', 'Lucky', 'Adedoyin',
#         'Etipou', 'Olumide', 'Olayinka', 'Regina', 'Alexander', 'Idris', 'Adebayo', 'Benson',
#         'Totosi', 'Gabriel', 'Anyaokei', 'Sandra', 'Samson', 'Arinze', 'Uduak', 'Sulaimon',
#         'Nelson', 'Adetayo', 'Daniel', 'Obiageli', 'Selimat', 'Elizabeth', 'Rachael', 'Saheed',
#         'Sanusi', 'Akintunde', 'Moshood', 'Emy', 'Ogechukwu', 'Adija', 'Ayodele D', 'Fabeyo',
#         'Godwin', 'Lois', 'Olabimpe', 'Funke', 'Akan', 'Vivian', 'Agozue', 'Helen', 'Owolabi',
#         'Kellywealth', 'Isioma', 'Titilayo', 'Felicia', 'Ella', 'Adesewa', 'Afolashade-O', 'Justin',
#         'Happiness', 'Sophia', 'Lilian', 'Fatima', 'Queen', 'Ifeoma', 'Ojuroye', 'Busayo'
#     ],
#     'Days Requested': [
#         11, 10, 10, 22, 20, 10, 12, 7, 12, 17, 10, 12, 15, 12, 10, 10, 12, 12, 10, 17, 
#         25, 13, 16, 12, 10, 15, 15, 10, 12, 22, 12, 11, 10, 15, 17, 22, 10, 20, 10, 10, 12,
#         17, 17, 12, 10, 10, 10, 22, 22, 10, 12, 20, 10, 11, 10, 10, 14, 10, 22, 15, 12, 10,
#         10, 10, 12, 17, 10, 12, 11, 10, 10, 10, 11, 10, 15, 15, 10, 10, 12, 12, 10, 10, 10,
#         20, 10, 5, 5, 15, 10, 10, 10, 10, 12, 10, 10, 15, 13, 12, 12, 22, 5, 12, 5
#     ]
# }

# # Create a DataFrame
# df = pd.DataFrame(data)

# # Save the DataFrame to a CSV file
# df.to_csv('days_requested.csv', index=False)

# # Optional: Print a message to confirm the file has been saved
# print("Data has been saved to days_requested.csv")

# # Plot the data
# plt.figure(figsize=(12, 8))
# plt.bar(df['Name'], df['Days Requested'])
# plt.xlabel('Name')
# plt.ylabel('Days Requested')
# plt.title('Days Requested by Each Employee')
# plt.xticks(rotation=90)  # Rotate the x-axis labels for better visibility
# plt.tight_layout()
# plt.show()








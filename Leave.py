


# import streamlit as st
# st.write("Text before the line")

# st.divider()  # Adds a horizontal line






# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.linear_model import LogisticRegression
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score

# # Larger Dataset with More Features
# data = {
#     'Name': ['John', 'Jane', 'Bill', 'Anna', 'Tom', 'Lisa', 'Gary', 'Nina', 'Mike', 'Sara',
#              'Robert', 'Lucy', 'Steve', 'Emma', 'Chris', 'Sophia', 'Brian', 'Olivia', 'Mark', 'Chloe'],
#     'Age': [18, 19, 20, 21, 22, 23, 21, 19, 20, 18, 21, 22, 20, 19, 18, 22, 21, 20, 19, 23],
#     'Score': [85, 78, 60, 90, 55, 95, 70, 88, 80, 77, 65, 92, 75, 85, 83, 55, 89, 76, 90, 72],
#     'Study Hours': [10, 9, 4, 12, 3, 14, 5, 11, 8, 7, 6, 13, 7, 10, 9, 3, 12, 6, 11, 5],
#     'Attendance': [90, 85, 60, 95, 50, 98, 65, 92, 80, 75, 68, 96, 72, 88, 87, 55, 93, 70, 94, 67],
#     'Test Difficulty': [3, 4, 5, 2, 5, 1, 4, 2, 3, 3, 4, 1, 3, 3, 2, 5, 2, 4, 1, 4],
#     'Passed': ['Yes', 'Yes', 'No', 'Yes', 'No', 'Yes', 'No', 'Yes', 'Yes', 'Yes',
#                'No', 'Yes', 'No', 'Yes', 'Yes', 'No', 'Yes', 'Yes', 'Yes', 'No']
# }

# # Create DataFrame
# df = pd.DataFrame(data)

# # Convert 'Passed' column to binary (Yes = 1, No = 0)
# df['Passed'] = df['Passed'].map({'Yes': 1, 'No': 0})

# # Features & Target
# X = df[['Age', 'Score', 'Study Hours', 'Attendance', 'Test Difficulty']]
# y = df['Passed']

# # Standardize the features
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)

# # Train/Test Split (80% train, 20% test)
# X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# # Initialize Models
# models = {
#     "Logistic Regression": LogisticRegression(),
#     "Decision Tree": DecisionTreeClassifier(),
#     "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
# }

# # Train and Evaluate Models
# accuracy_results = {}

# for name, model in models.items():
#     model.fit(X_train, y_train)
#     y_pred = model.predict(X_test)
#     accuracy = accuracy_score(y_test, y_pred)
#     accuracy_results[name] = accuracy
#     print(f"{name} Accuracy: {accuracy:.2f}")

# # Compare Model Accuracy
# plt.figure(figsize=(8, 5))
# plt.bar(accuracy_results.keys(), accuracy_results.values(), color=['blue', 'green', 'red'])
# plt.ylabel("Accuracy Score")
# plt.title("Model Accuracy Comparison")
# plt.ylim(0, 3)  # Keep the scale from 0 to 1
# plt.show()

# # Best Model
# best_model = max(accuracy_results, key=accuracy_results.get)
# print(f"🏆 Best Model: {best_model} with Accuracy {accuracy_results[best_model]:.2f}")






# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.tree import DecisionTreeClassifier, plot_tree
# from sklearn.preprocessing import LabelEncoder
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score, classification_report

# # Load dataset
# data = {
#     "DATE": ["1-Feb", "1-Feb", "1-Feb", "1-Feb", "1-Feb", "3-Feb", "3-Feb", "3-Feb", "3-Feb", "4-Feb", "4-Feb", "5-Feb"],
#     "USERS": [5636120, 21391508, 14705517, 3603622, 999565, 9219911, 20768625, 2951010, 22354357, 1530444, 22361571, 18220873],
#     "REQUEST MEDIUM": ["SM(TWITTER)", "BM", "BM", "BM", "BM", "BM", "SM(IG)", "phone call", "BM", "SM(TWITTER)", "phone call", "BM"],
#     "USER REASON": [
#         "Excluded himself to control his gambling behaviour", "User claims he needed to take a short break",
#         "User claims he needed to take a short break", "User claims he needed to take a short break",
#         "User claims he needed to take a short break", "User claims he needed to take a short break",
#         "User claims he needed to take a short break", "Needed to limit his spending on betting",
#         "User claims he needed to take a short break", "Excluded himself to control his gambling behaviour",
#         "User claims he needed to take a short break", "User claims he needed to take a short break"
#     ],
#     "NUMBER OF TIME": [2, 1, 2, 1, 1, 1, 2, 1, 1, 2, 2, 1],
#     "SELF-EXCLUDED PERIOD": [2, 13, 12, 11, 10, 4, 5, 5, 4, 2, 4, 14],
#     "OBSERVATION RATING": ["Low", "Low", "Low", "Low", "Low", "Low", "Low", "Medium", "Low", "Medium", "Low", "Low"],
# }

# df = pd.DataFrame(data)

# # Encode categorical features
# encoder = LabelEncoder()
# df["REQUEST MEDIUM"] = encoder.fit_transform(df["REQUEST MEDIUM"])
# df["USER REASON"] = encoder.fit_transform(df["USER REASON"])
# df["OBSERVATION RATING"] = encoder.fit_transform(df["OBSERVATION RATING"])  # Target variable

# # Define features and target
# X = df[["USERS", "REQUEST MEDIUM", "USER REASON", "NUMBER OF TIME", "SELF-EXCLUDED PERIOD"]]
# y = df["OBSERVATION RATING"]

# # Split the data
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Train the model
# clf = DecisionTreeClassifier(max_depth=3, random_state=42)
# clf.fit(X_train, y_train)

# # Visualizing the Decision Tree
# plt.figure(figsize=(12, 6))
# plot_tree(clf, filled=True, feature_names=X.columns, class_names=["Low", "Medium"])
# plt.show()





# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder
# from sklearn.linear_model import LogisticRegression
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score

# # File path (Update the path accordingly)
# file_path = "monthly-analysis.csv"  # Change this to your actual file path

# # Read dataset
# df = pd.read_csv(file_path)

# # Ensure correct column names (strip any accidental spaces)
# df.columns = df.columns.str.strip()

# # Encode categorical variables
# label_enc = LabelEncoder()
# df["REQUEST MEDIUM"] = label_enc.fit_transform(df["REQUEST MEDIUM"])
# df["USER REASON"] = label_enc.fit_transform(df["USER REASON"])
# df["OBSERVATION RATING"] = label_enc.fit_transform(df["OBSERVATION RATING"])

# # Define target variable: Addicted (1 if NUMBER OF TIME >= 2, else 0)
# df["ADDICTED"] = (df["NUMBER OF TIME"] >= 2).astype(int)

# # Define features (X) and target (y)
# X = df[["NUMBER OF TIME", "SELF-EXCLUDED PERIOD", "OBSERVATION RATING", "REQUEST MEDIUM", "USER REASON"]]
# y = df["ADDICTED"]

# # Split data into training and test sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Define models
# models = {
#     "Logistic Regression": LogisticRegression(),
#     "Decision Tree": DecisionTreeClassifier(),
#     "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
# }

# # Train and evaluate each model
# for name, model in models.items():
#     model.fit(X_train, y_train)
#     y_pred = model.predict(X_test)
#     accuracy = accuracy_score(y_test, y_pred)
#     print(f"{name} Accuracy: {accuracy:.2f}")

# # Identify Addicted Users (NUMBER OF TIME >= 2 and OBSERVATION RATING == Medium)
# medium_rating = label_enc.transform(["Medium"])[0] if "Medium" in label_enc.classes_ else None
# if medium_rating is not None:
#     addicted_users = df[(df["NUMBER OF TIME"] >= 2) & (df["OBSERVATION RATING"] == medium_rating)]
#     print("\nAddicted Users:")
#     print(addicted_users if not addicted_users.empty else "No addicted users found.")
# else:
#     print("Warning: 'Medium' rating not found in dataset.")

#     # Save output to a text file
# output_file = "addicted_users.txt"
# with open(output_file, "w", encoding="utf-8") as f:
#     f.write("Addicted Users:\n")
#     f.write(addicted_users.to_string(index=False))

# print(f"\nAddicted users saved to '{output_file}'")














# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder
# from sklearn.linear_model import LogisticRegression
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score

# # Dataset
# data = {
#     "DATE": ["1-Feb", "1-Feb", "1-Feb", "1-Feb", "1-Feb", "3-Feb", "3-Feb", "3-Feb", "3-Feb", "4-Feb", "4-Feb", "5-Feb"],
#     "USERS": [5636120, 21391508, 14705517, 3603622, 999565, 9219911, 20768625, 2951010, 22354357, 1530444, 22361571, 18220873],
#     "REQUEST MEDIUM": ["SM(TWITTER)", "BM", "BM", "BM", "BM", "BM", "SM(IG)", "phone call", "BM", "SM(TWITTER)", "phone call", "BM"],
#     "USER REASON": [
#         "Excluded himself to control his gambling behaviour", "User claims he needed to take a short break",
#         "User claims he needed to take a short break", "User claims he needed to take a short break",
#         "User claims he needed to take a short break", "User claims he needed to take a short break",
#         "User claims he needed to take a short break", "Needed to limit his spending on betting",
#         "User claims he needed to take a short break", "Excluded himself to control his gambling behaviour",
#         "User claims he needed to take a short break", "User claims he needed to take a short break"
#     ],
#     "NUMBER OF TIME": [2, 1, 2, 1, 1, 1, 2, 1, 1, 2, 2, 1],
#     "SELF-EXCLUDED PERIOD": [2, 13, 12, 11, 10, 4, 5, 5, 4, 2, 4, 14],
#     "OBSERVATION RATING": ["Low", "Low", "Low", "Low", "Low", "Low", "Low", "Medium", "Low", "Medium", "Low", "Low"],
# }

# # Convert to DataFrame
# df = pd.DataFrame(data)

# # Encode categorical variables
# label_enc = LabelEncoder()
# df["REQUEST MEDIUM"] = label_enc.fit_transform(df["REQUEST MEDIUM"])
# df["USER REASON"] = label_enc.fit_transform(df["USER REASON"])
# df["OBSERVATION RATING"] = label_enc.fit_transform(df["OBSERVATION RATING"])

# # Define target variable: Addicted (1 if NUMBER OF TIME >= 2, else 0)
# df["ADDICTED"] = (df["NUMBER OF TIME"] >= 2).astype(int)

# # Define features (X) and target (y)
# X = df[["NUMBER OF TIME", "SELF-EXCLUDED PERIOD", "OBSERVATION RATING", "REQUEST MEDIUM", "USER REASON"]]
# y = df["ADDICTED"]

# # Split data into training and test sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Define models
# models = {
#     "Logistic Regression": LogisticRegression(),
#     "Decision Tree": DecisionTreeClassifier(),
#     "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
# }

# # Train and evaluate each model
# for name, model in models.items():
#     model.fit(X_train, y_train)
#     y_pred = model.predict(X_test)
#     accuracy = accuracy_score(y_test, y_pred)
#     print(f"{name} Accuracy: {accuracy:.2f}")

# # Identify Addicted Users (NUMBER OF TIME >= 2 and OBSERVATION RATING == Medium)
# addicted_users = df[(df["NUMBER OF TIME"] >= 2) & (df["OBSERVATION RATING"] == label_enc.transform(["Medium"])[0])]

# # Print addicted users
# print("\nAddicted Users:")
# print(addicted_users)
















# import pandas as pd
# import numpy as np
# import seaborn as sns
# import matplotlib.pyplot as plt
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import train_test_split, GridSearchCV
# from sklearn.metrics import accuracy_score, classification_report

# # Sample data
# data = {
#     "DATE": ["1-Feb", "1-Feb", "1-Feb", "1-Feb", "1-Feb", "3-Feb", "3-Feb", "3-Feb", "3-Feb", "4-Feb", "4-Feb", "5-Feb"],
#     "USERS": [5636120, 21391508, 14705517, 3603622, 999565, 9219911, 20768625, 2951010, 22354357, 1530444, 22361571, 18220873],
#     "REQUEST MEDIUM": ["SM(TWITTER)", "BM", "BM", "BM", "BM", "BM", "SM(IG)", "phone call", "BM", "SM(TWITTER)", "phone call", "BM"],
#     "USER REASON": [
#         "Excluded himself to control his gambling behaviour", "User claims he needed to take a short break",
#         "User claims he needed to take a short break", "User claims he needed to take a short break",
#         "User claims he needed to take a short break", "User claims he needed to take a short break",
#         "User claims he needed to take a short break", "Needed to limit his spending on betting",
#         "User claims he needed to take a short break", "Excluded himself to control his gambling behaviour",
#         "User claims he needed to take a short break", "User claims he needed to take a short break"
#     ],
#     "NUMBER OF TIME": [2, 1, 2, 1, 1, 1, 2, 1, 1, 2, 2, 1],
#     "SELF-EXCLUDED PERIOD": [2, 13, 12, 11, 10, 4, 5, 5, 4, 2, 4, 14],  # Converted to numeric days
#     "OBSERVATION RATING": ["Low", "Low", "Low", "Low", "Low", "Low", "Low", "Medium", "Low", "Medium", "Low", "Low"],
#     "PHONE": ["09068384470", "07089616894", "07013411884", "08095072279", "08160593778", "09043535907", "08120577068", "09037993944", "08075822390", "07038358925", "07019249045", "08166424828"],
#     "STATE": ["OYO", "BENUE", "DELTA", "BAUCHI", "RIVERS", "LAGOS", "AKWA IBOM", "LAGOS", "EDO", "ABUJA", "LAGOS", "IMO"]
# }

# # Convert to DataFrame
# df = pd.DataFrame(data)

# # Convert DATE to datetime
# df["DATE"] = pd.to_datetime(df["DATE"], format="%d-%b")

# # Convert observation rating to numeric values
# df["Rating_Numeric"] = df["OBSERVATION RATING"].map({"Medium": 1, "Low": 0})

# # Convert REQUEST MEDIUM to categorical values
# df["REQUEST_MEDIUM_NUMERIC"] = df["REQUEST MEDIUM"].astype("category").cat.codes

# # Select features and target variable
# X = df[["NUMBER OF TIME", "SELF-EXCLUDED PERIOD", "USERS", "REQUEST_MEDIUM_NUMERIC"]]
# y = df["Rating_Numeric"]

# # Split into training & testing data
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# # Hyperparameter tuning for Random Forest
# param_grid = {"n_estimators": [50, 100, 200], "max_depth": [None, 5, 10]}
# grid = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5, scoring="accuracy")
# grid.fit(X_train, y_train)

# # Best model
# best_model = grid.best_estimator_
# y_pred = best_model.predict(X_test)

# # Model Accuracy
# accuracy = accuracy_score(y_test, y_pred)
# print(f"🔍 Model Accuracy: {accuracy * 100:.2f}%\n")

# # Classification Report
# print("📊 Classification Report:\n", classification_report(y_test, y_pred))

# # Analyzing misclassified cases
# X_test_copy = X_test.copy()
# X_test_copy["Actual Rating"] = y_test.values
# X_test_copy["Predicted Rating"] = y_pred
# misclassified = X_test_copy[X_test_copy["Actual Rating"] != X_test_copy["Predicted Rating"]]

# # Display misclassified cases
# print("⚠️ Misclassified Cases:\n", misclassified)

# # Visualization: Feature Importance
# plt.figure(figsize=(8, 5))
# sns.barplot(x=best_model.feature_importances_, y=X.columns)
# plt.title("Feature Importance in Self-Exclusion Prediction")
# plt.xlabel("Importance")
# plt.ylabel("Feature")
# plt.show()









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
# required_columns = ["OBSERVATION RATING", "NUMBER OF TIME", "USERS", "USER REASON", "PHONE", "REQUEST MEDIUM", "STATE", "DATE"]
# missing_columns = [col for col in required_columns if col not in df.columns]
# if missing_columns:
#     print(f"Error: Missing columns {missing_columns} in the data.")
#     exit()

# # Ensure phone numbers are treated as strings and preserve leading zeros
# df['PHONE'] = df['PHONE'].astype(str).str.zfill(11)

# # Total Requests
# total_requests = len(df)

# # Standardize the 'OBSERVATION RATING' column
# df["OBSERVATION RATING"] = df["OBSERVATION RATING"].str.lower()

# # Observation Rating Counts
# rating_counts = df["OBSERVATION RATING"].value_counts()

# # Filter Low and Medium observation ratings
# low_count_filtered = df[df["OBSERVATION RATING"] == 'low'][["USERS", "USER REASON", "NUMBER OF TIME", "PHONE"]]
# medium_count_filtered = df[df["OBSERVATION RATING"] == 'medium'][["USERS", "USER REASON", "NUMBER OF TIME", "PHONE"]]

# # Group by "REQUEST MEDIUM"
# grouped_medium = df.groupby("REQUEST MEDIUM").size().reset_index(name="Occurrence").sort_values(by="Occurrence", ascending=False)

# # Group by "USER REASON"
# grouped_by_reason = df.groupby("USER REASON").size().reset_index(name="Count").sort_values(by="Count", ascending=False)

# # Count Self-Excluded Employees by State & Include Employee_IDs
# state_counts = df.groupby("STATE").agg(
#     Count=("NUMBER OF TIME", "count"), 
#     USERS=("USERS", lambda x: list(x))
# ).reset_index()

# # Sort in descending order
# state_counts = state_counts.sort_values(by="Count", ascending=False)

# # Convert USERS list to string
# state_counts["USERS"] = state_counts["USERS"].apply(lambda x: ", ".join(map(str, x)))

# # Convert "DATE" column to datetime format
# df["DATE"] = pd.to_datetime(df["DATE"])
# daily_resolutions = df.groupby(df["DATE"].dt.date).size().reset_index(name="Resolutions Count").sort_values(by="Resolutions Count", ascending=False)

# # Convert rating_counts into a DataFrame
# rating_counts_df = df.pivot_table(index="STATE", columns="OBSERVATION RATING", values="NUMBER OF TIME", aggfunc="sum", fill_value=0)

# # Save results to TXT
# output_txt = "monthly-analysis-output.txt"
# with open(output_txt, "w") as f:
#     f.write(f"Total self-exclusion requests for the month: {total_requests}\n\n")
#     f.write("Observation Rating Counts:\n")
#     f.write(rating_counts.to_string() + "\n\n")
#     f.write("Requests by Medium:\n")
#     f.write(grouped_medium.to_string(index=False) + "\n\n")
#     f.write("Requests by User Reason:\n")
#     f.write(grouped_by_reason.to_string(index=False) + "\n\n")
#     f.write("State Counts:\n")
#     f.write(state_counts.to_string(index=False) + "\n\n")
#     f.write("Daily Resolutions:\n")
#     f.write(daily_resolutions.to_string(index=False) + "\n")

# # Generate and save graphs
# graph_files = []

# # Observation Rating Bar Chart
# plt.figure(figsize=(8, 6))
# sns.barplot(x=rating_counts.index, y=rating_counts.values, palette="viridis")
# plt.title("Observation Rating Counts")
# plt.xlabel("Observation Rating")
# plt.ylabel("Count")
# bar_chart_file = "observation_rating_counts.png"
# plt.savefig(bar_chart_file)
# graph_files.append(bar_chart_file)
# plt.close()

# # Users Count by State Bar Chart
# plt.figure(figsize=(8, 6))
# sns.barplot(x="STATE", y="Count", data=state_counts, palette="viridis")
# plt.title("Users count by state")
# plt.xlabel("State")
# plt.ylabel("Number of Users")
# state_chart_file = "state_counts.png"
# plt.savefig(state_chart_file)
# graph_files.append(state_chart_file)
# plt.close()

# # Request Medium Pie Chart
# plt.figure(figsize=(8, 8))
# plt.pie(
#     grouped_medium["Occurrence"],
#     labels=grouped_medium["REQUEST MEDIUM"],
#     autopct="%1.1f%%",
#     startangle=140,
#     colors=sns.color_palette("pastel"),
# )
# plt.title("Distribution of Request Medium")
# pie_chart_file = "request_medium_distribution.png"
# plt.savefig(pie_chart_file)
# graph_files.append(pie_chart_file)
# plt.close()

# # User Reasons Bar Chart
# plt.figure(figsize=(10, 6))
# sns.barplot(x=grouped_by_reason["Count"], y=grouped_by_reason["USER REASON"], palette="coolwarm")
# plt.title("Top User Reasons")
# plt.xlabel("Count")
# plt.ylabel("User Reason")
# user_reason_chart_file = "top_user_reasons.png"
# plt.savefig(user_reason_chart_file)
# graph_files.append(user_reason_chart_file)
# plt.close()

# # Line Graph for Resolutions Over Time
# plt.figure(figsize=(10, 6))
# sns.lineplot(x=daily_resolutions["DATE"], y=daily_resolutions["Resolutions Count"], marker="o", color="blue")
# plt.title("Self-Exclusion Resolutions Per Day")
# plt.xlabel("Date")
# plt.ylabel("Resolution Count")
# plt.xticks(rotation=45)
# plt.grid(True)
# line_chart_file = "self_exclusion_resolutions.png"
# plt.savefig(line_chart_file)
# graph_files.append(line_chart_file)
# plt.close()

# # Heatmap Visualization
# plt.figure(figsize=(8, 5))
# sns.heatmap(rating_counts_df, annot=True, cmap="coolwarm", linewidths=0.5, fmt="d")
# plt.title("Self-Exclusion Ratings Heatmap")
# plt.xlabel("Observation Rating")
# plt.ylabel("State")
# heatmap_file = "ratings_heatmap.png"
# plt.savefig(heatmap_file)
# graph_files.append(heatmap_file)
# plt.close()







# import pandas as pd

# # Sample data (employee self-exclusion records)
# data = {
#     "Employee_ID": [101, 102, 103, 104, 105, 106, 107, 108, 109, 110],
#     "State": ["Lagos", "Abuja", "Lagos", "Rivers", "Abuja", "Lagos", "Kaduna", "Abuja", "Ogun", "Rivers"],
#     "Self_Excluded": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  # 1 means self-excluded
# }

# # Convert to DataFrame
# df = pd.DataFrame(data)

# # Count self-excluded employees and include Employee_IDs
# state_counts = df.groupby("State").agg(
#     Count=("Self_Excluded", "count"), 
#     Employee_ID=("Employee_ID", lambda x: list(x))
# ).reset_index()

# # Convert Employee_ID list to string for better display
# state_counts["Employee_ID"] = state_counts["Employee_ID"].apply(lambda x: ", ".join(map(str, x)))

# # Add a Rating column based on Self-Excluded Count
# state_counts["Rating"] = state_counts["Count"].apply(lambda x: "Medium" if x >= 2 else "Low")

# # Convert Rating to numeric for correlation calculation
# state_counts["Rating_Numeric"] = state_counts["Rating"].map({"Medium": 1, "Low": 0})

# # Compute correlation between Self-Exclusion Count and Rating_Numeric
# correlation = state_counts["Count"].corr(state_counts["Rating_Numeric"])

# print(f"Correlation: {correlation}")
# print()
# print()

# # Sort in descending order
# state_counts = state_counts.sort_values(by="Count", ascending=False)

# # Get the top 3 and bottom 3 states
# top_3_states = state_counts.nlargest(3, 'Count')
# bottom_3_states = state_counts.nsmallest(3, 'Count')



# # Print the results
# print(f"🏆 Top 3 states with highest self-exclusion: {top_3_states}")
# print()
# print()

# print(f"⬇️ Bottom 3 states with lowest self-exclusion: {bottom_3_states}")













# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt

# # Sample data (based on provided dataset)
# data = {
#     "DATE": ["1-Feb", "1-Feb", "1-Feb", "1-Feb", "1-Feb", "3-Feb", "3-Feb", "3-Feb", "3-Feb", "4-Feb", "4-Feb", "5-Feb"],
#     "USERS": [5636120, 21391508, 14705517, 3603622, 999565, 9219911, 20768625, 2951010, 22354357, 1530444, 22361571, 18220873],
#     "REQUEST MEDIUM": ["SM(TWITTER)", "BM", "BM", "BM", "BM", "BM", "SM(IG)", "phone call", "BM", "SM(TWITTER)", "phone call", "BM"],
#     "USER REASON": [
#         "Excluded himself to control his gambling behaviour", "User claims he needed to take a short break",
#         "User claims he needed to take a short break", "User claims he needed to take a short break",
#         "User claims he needed to take a short break", "User claims he needed to take a short break",
#         "User claims he needed to take a short break", "Needed to limit his spending on betting",
#         "User claims he needed to take a short break", "Excluded himself to control his gambling behaviour",
#         "User claims he needed to take a short break", "User claims he needed to take a short break"
#     ],
#     "NUMBER OF TIME": [2, 1, 2, 1, 1, 1, 2, 1, 1, 2, 2, 1],
#     "SELF-EXCLUDED PERIOD": ["2 Days", "13 Days", "12 Days", "11 Days", "10 Days", "4 Days", "5 Days", "5 Days", "4 Days", "2 Days", "4 Days", "14 Days"],
#     "OBSERVATION RATING": ["Low", "Low", "Low", "Low", "Low", "Low", "Low", "Medium", "Low", "Medium", "Low", "Low"],
#     "PHONE": ["09068384470", "07089616894", "07013411884", "08095072279", "08160593778", "09043535907", "08120577068", "09037993944", "08075822390", "07038358925", "07019249045", "08166424828"],
#     "STATE": ["OYO", "BENUE", "DELTA", "BAUCHI", "RIVERS", "LAGOS", "AKWA IBOM", "LAGOS", "EDO", "ABUJA", "LAGOS", "IMO"]
# }

# # Convert to DataFrame
# df = pd.DataFrame(data)

# # Count occurrences of ratings per state
# rating_counts = df.pivot_table(index="STATE", columns="OBSERVATION RATING", aggfunc="size", fill_value=0)

# print(rating_counts)
# print()

# # Add total column for sorting
# rating_counts["Total"] = rating_counts.sum(axis=1)

# # Sort states by total self-exclusion requests
# rating_counts = rating_counts.sort_values(by="Total", ascending=False)

# print(rating_counts)
# print()


# rating_counts["% Low"] = (rating_counts["Low"] / rating_counts["Total"] * 100).round(1)
# rating_counts["% Medium"] = (rating_counts["Medium"] / rating_counts["Total"] * 100).round(1)

# print(rating_counts)


# # Get states with highest and lowest 'Medium' ratings
# most_medium_state = rating_counts["Medium"].idxmax()
# least_medium_state = rating_counts["Medium"].idxmin()

# # Get states with highest and lowest 'Low' ratings
# most_low_state = rating_counts["Low"].idxmax()
# least_low_state = rating_counts["Low"].idxmin()

# # Display results in Streamlit
# # st.title("Self-Exclusion Analysis")

# print(f"**State with Most 'Medium' Ratings:** {most_medium_state} ({rating_counts.loc[most_medium_state, 'Medium']} occurrences)")
# print(f"**State with Least 'Medium' Ratings:** {least_medium_state} ({rating_counts.loc[least_medium_state, 'Medium']} occurrences)")
# print(f"**State with Most 'Low' Ratings:** {most_low_state} ({rating_counts.loc[most_low_state, 'Low']} occurrences)")
# print(f"**State with Least 'Low' Ratings:** {least_low_state} ({rating_counts.loc[least_low_state, 'Low']} occurrences)")

# # Plot Medium and Low Ratings by State


# # Plot Medium and Low Ratings by State
# plt.subplots(figsize=(10, 6))
# rating_counts.plot(kind='bar', stacked=True, colormap="coolwarm")
# plt.title("Observation Ratings by State")
# plt.xlabel("State")
# plt.ylabel("Count")
# plt.xticks(rotation=45)
# plt.show()


# ### 🔹 Bar Chart – Ratings per State
# plt.figure(figsize=(10, 6))
# rating_counts.drop(columns=["Total"]).plot(kind="bar", stacked=True, colormap="viridis", figsize=(10, 6))

# plt.title("Self-Exclusion Ratings per State", fontsize=14)
# plt.xlabel("State", fontsize=12)
# plt.ylabel("Count", fontsize=12)
# plt.xticks(rotation=45)
# plt.legend(title="Observation Rating")
# plt.grid(axis="y", linestyle="--", alpha=0.7)
# plt.show()

# ### 🔹 Pie Chart – Overall Ratings Distribution
# rating_totals = rating_counts.drop(columns=["Total"]).sum()

# plt.figure(figsize=(8, 6))
# plt.pie(rating_totals, labels=rating_totals.index, autopct="%1.1f%%", colors=["#ff9999", "#66b3ff"], startangle=140)
# plt.title("Overall Distribution of Self-Exclusion Ratings")
# plt.show()

# ### 🔹 Heatmap – Ratings Across States
# plt.figure(figsize=(8, 5))
# sns.heatmap(rating_counts.drop(columns=["Total"]), annot=True, cmap="coolwarm", linewidths=0.5, fmt="d")

# plt.title("Self-Exclusion Ratings Heatmap", fontsize=14)
# plt.xlabel("Observation Rating", fontsize=12)
# plt.ylabel("State", fontsize=12)
# plt.show()

















# import pandas as pd
# import streamlit as st
# import seaborn as sns
# import matplotlib.pyplot as plt

# # Sample data (employee self-exclusion records)
# data = {
#     "Employee_ID": [101, 102, 103, 104, 105, 106, 107, 108, 109, 110],
#     "State": ["Lagos", "Abuja", "Lagos", "Rivers", "Abuja", "Lagos", "Kaduna", "Abuja", "Ogun", "Rivers"],
#     "Self_Excluded": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  # 1 means self-excluded
# }

# # Convert to DataFrame
# df = pd.DataFrame(data)

# # Count self-excluded employees and include Employee_IDs
# state_counts = df.groupby("State").agg(
#     Count=("Self_Excluded", "count"), 
#     Employee_ID=("Employee_ID", lambda x: list(x))
# ).reset_index()

# # Sort in descending order
# state_counts = state_counts.sort_values(by="Count", ascending=False)

# # Convert Employee_ID list to string for better display in Streamlit
# state_counts["Employee_ID"] = state_counts["Employee_ID"].apply(lambda x: ", ".join(map(str, x)))

# #Get the top 3 and bottom 3 states
# top_3_states = state_counts.nlargest(3, 'Count')
# bottom_3_states = state_counts.nsmallest(3, 'Count')

# # Streamlit App
# st.title("Self-Exclusion Count by State")

# # Display DataFrame
# st.write("### Self-Exclusion Count by State (Sorted in Descending Order)")
# st.dataframe(state_counts)


# st.subheader("Top 3 and Bottom 3 State")

# col1, col2 = st.columns(2)

# col1.write(top_3_states)
# col2.write(bottom_3_states)



# # Display Top 3 States
# st.write("### Top 3 States with Highest Self-Exclusion")
# st.dataframe(top_3_states)

# # Display Bottom 3 States
# st.write("### Bottom 3 States with Lowest Self-Exclusion")
# st.dataframe(bottom_3_states)

# Seaborn Bar Chart for All States
# st.write("### Self-Exclusion Distribution by State")
# plt.figure(figsize=(10, 5))
# sns.barplot(x="State", y="Count", data=state_counts, palette="viridis")
# plt.xlabel("State")
# plt.ylabel("Number of Self-Excluded Employees")
# plt.title("Self-Exclusion Count by State")
# plt.xticks(rotation=45)
# st.pyplot(plt)

# # Seaborn Bar Chart for Top 3 States
# st.write("### Top 3 States with Highest Self-Exclusion")
# plt.figure(figsize=(8, 4))
# sns.barplot(x="State", y="Count", data=top_3_states, palette="Reds_r")
# plt.xlabel("State")
# plt.ylabel("Number of Self-Excluded Employees")
# plt.title("Top 3 States with Highest Self-Exclusion")
# plt.xticks(rotation=45)
# st.pyplot(plt)

# # Seaborn Bar Chart for Bottom 3 States
# st.write("### Bottom 3 States with Lowest Self-Exclusion")
# plt.figure(figsize=(8, 4))
# sns.barplot(x="State", y="Count", data=bottom_3_states, palette="Blues_r")
# plt.xlabel("State")
# plt.ylabel("Number of Self-Excluded Employees")
# plt.title("Bottom 3 States with Lowest Self-Exclusion")
# plt.xticks(rotation=45)
# st.pyplot(plt)

# combined_states = pd.concat([top_3_states, bottom_3_states])


# # Seaborn Bar Chart for Combined States
# st.write("### Self-Exclusion in Top 3 and Bottom 3 States")
# plt.figure(figsize=(8, 4))
# sns.barplot(x="State", y="Count", data=combined_states, palette="coolwarm")
# plt.xlabel("State")
# plt.ylabel("Number of Self-Excluded Employees")
# plt.title("Self-Exclusion Count (Top 3 and Bottom 3 States)")
# plt.xticks(rotation=45)
# st.pyplot(plt)








# # WEEKLY CALL Report

# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns

# # Sample data (replace with your actual dataset)
# data = {
#     "Department": ["Customer Service Dept", "Commercial Dept"],
#     "Success_Ratio": [96.43, 90.00],
#     "Call_Drop_Rate": [30.25, 68.75],
#     "FCR_Rate": [67.24, 28.12]
# }

# # Create DataFrame
# df = pd.DataFrame(data)

# # Visualization: Bar Graph comparing key metrics (Success Ratio, Call Drop Rate, FCR Rate)
# fig, ax = plt.subplots(1, 3, figsize=(18, 6))

# # Success Ratio
# sns.barplot(x=df["Department"], y=df["Success_Ratio"], ax=ax[0], palette="Blues")
# ax[0].set_title("Success Ratio by Department")
# ax[0].set_ylabel("Success Ratio (%)")

# # Call Drop Rate
# sns.barplot(x=df["Department"], y=df["Call_Drop_Rate"], ax=ax[1], palette="Reds")
# ax[1].set_title("Call Drop Rate by Department")
# ax[1].set_ylabel("Call Drop Rate (%)")

# # FCR Rate
# sns.barplot(x=df["Department"], y=df["FCR_Rate"], ax=ax[2], palette="Greens")
# ax[2].set_title("First Call Resolution Rate by Department")
# ax[2].set_ylabel("FCR Rate (%)")

# # Adjust layout and display the graph
# plt.tight_layout()
# plt.show()
















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
# required_columns = ["OBSERVATION RATING", "NUMBER OF TIME", "USERS", "USER REASON", "PHONE", "REQUEST MEDIUM", "STATE"]
# missing_columns = [col for col in required_columns if col not in df.columns]
# if missing_columns:
#     print(f"Error: Missing columns {missing_columns} in the data.")
#     exit()

# # Ensure phone numbers are treated as strings and preserve leading zeros
# df['PHONE'] = df['PHONE'].astype(str).str.zfill(11)

# # Total Requests
# total_requests = len(df)

# # Standardize the 'OBSERVATION RATING' column to lowercase
# df["OBSERVATION RATING"] = df["OBSERVATION RATING"].str.lower()

# # Observation Rating Counts
# rating_counts = df["OBSERVATION RATING"].value_counts()

# # Group by "REQUEST MEDIUM"
# grouped_medium = df.groupby("REQUEST MEDIUM").size().reset_index(name="Occurrence").sort_values(by="Occurrence", ascending=False)

# # Group by "USER REASON"
# grouped_by_reason = df.groupby("USER REASON").size().reset_index(name="Count").sort_values(by="Count", ascending=False)

# # Count Self-Excluded Employees by State & Include USERS
# state_counts = df.groupby("STATE").agg(
#     Count=("NUMBER OF TIME", "count"), 
#     USERS=("USERS", lambda x: list(x))
# ).reset_index()
# state_counts = state_counts.sort_values(by="Count", ascending=False)
# state_counts["USERS"] = state_counts["USERS"].apply(lambda x: ", ".join(map(str, x)))

# # Save results to CSV and TXT
# output_csv = "monthly-analysis-output.csv"
# output_txt = "monthly-analysis-output.txt"
# df.to_csv(output_csv, index=False)

# with open(output_txt, "w") as file:
#     file.write(f"Total Self-Exclusion Requests: {total_requests}\n\n")
#     file.write("Observation Rating Counts:\n")
#     file.write(rating_counts.to_string() + "\n\n")
#     file.write("Request Medium Distribution:\n")
#     file.write(grouped_medium.to_string(index=False) + "\n\n")
#     file.write("User Reasons Count:\n")
#     file.write(grouped_by_reason.to_string(index=False) + "\n\n")
#     file.write("Self-Excluded Users by State:\n")
#     file.write(state_counts.to_string(index=False) + "\n\n")

# # Generate and save graphs
# graph_files = []

# # Observation Rating Bar Chart
# plt.figure(figsize=(8, 6))
# sns.barplot(x=rating_counts.index, y=rating_counts.values, palette="viridis")
# plt.title("Observation Rating Counts")
# plt.xlabel("Observation Rating")
# plt.ylabel("Count")
# plt.savefig("observation_rating_counts.png")
# graph_files.append("observation_rating_counts.png")
# plt.close()

# # Users Count by State
# plt.figure(figsize=(10, 5))
# sns.barplot(x="STATE", y="Count", data=state_counts, palette="viridis")
# plt.title("Users Count by State")
# plt.xlabel("State")
# plt.ylabel("Number of Users")
# plt.xticks(rotation=45)
# plt.savefig("state_counts.png")
# graph_files.append("state_counts.png")
# plt.close()

# # Request Medium Pie Chart
# plt.figure(figsize=(8, 8))
# plt.pie(
#     grouped_medium["Occurrence"],
#     labels=grouped_medium["REQUEST MEDIUM"],
#     autopct="%1.1f%%",
#     startangle=140,
#     colors=sns.color_palette("pastel"),
# )
# plt.title("Distribution of Request Medium")
# plt.savefig("request_medium_distribution.png")
# graph_files.append("request_medium_distribution.png")
# plt.close()

# # User Reasons Bar Chart
# plt.figure(figsize=(10, 6))
# sns.barplot(x=grouped_by_reason["Count"], y=grouped_by_reason["USER REASON"], palette="coolwarm")
# plt.title("Top User Reasons")
# plt.xlabel("Count")
# plt.ylabel("User Reason")
# plt.savefig("top_user_reasons.png")
# graph_files.append("top_user_reasons.png")
# plt.close()

# print(f"Analysis saved in {output_txt} and {output_csv}")












# import pandas as pd
# import streamlit as st
# import matplotlib.pyplot as plt
# import seaborn as sns

# # Sample leave application data
# data = {
#     'Employee': ['John', 'Sarah', 'Mike', 'Anna', 'David', 'Emily', 'Tom', 'Sophia', 'Chris', 'Emma'],
#     'Leave_Start': ['2024-01-10', '2024-02-15', '2024-02-20', '2024-03-05', '2024-03-18',
#                     '2024-04-25', '2024-04-30', '2024-05-10', '2024-02-25', '2024-04-12'],
#     'Leave_End': ['2024-01-12', '2024-02-18', '2024-02-22', '2024-03-07', '2024-03-22',
#                   '2024-04-28', '2024-05-02', '2024-05-12', '2024-02-28', '2024-04-15'],
#     'Leave_Type': ['Annual', 'Sick', 'Annual', 'Maternity', 'Casual', 'Sick', 'Annual', 'Casual', 'Sick', 'Annual']
# }

# # Convert to DataFrame
# leave_df = pd.DataFrame(data)

# # Convert date columns to datetime format
# leave_df['Leave_Start'] = pd.to_datetime(leave_df['Leave_Start'])
# leave_df['Leave_End'] = pd.to_datetime(leave_df['Leave_End'])

# # Calculate leave duration
# leave_df['Leave_Duration'] = (leave_df['Leave_End'] - leave_df['Leave_Start']).dt.days + 1

# # Extract Month Name from Leave_Start date
# leave_df['Month_Name'] = leave_df['Leave_Start'].dt.strftime('%B')

# # Define month order for sorting
# month_order = ['January', 'February', 'March', 'April', 'May', 'June', 
#                'July', 'August', 'September', 'October', 'November', 'December']

# # Convert Month_Name column to categorical
# leave_df['Month_Name'] = pd.Categorical(leave_df['Month_Name'], categories=month_order, ordered=True)

# # Streamlit app layout
# st.title("📊 Leave Analytics Dashboard")

# st.write("""
# This interactive dashboard provides insights into leave applications, 
# including trends, leave type distribution, and average leave durations.
# """)

# # Sidebar Filters
# st.sidebar.header("🔍 Filters")

# # Employee Filter
# selected_employee = st.sidebar.selectbox("Select Employee:", ["All"] + list(leave_df['Employee'].unique()))

# # Leave Type Filter
# selected_leave_types = st.sidebar.multiselect("Select Leave Type:", leave_df['Leave_Type'].unique(), default=leave_df['Leave_Type'].unique())

# # Date Range Filter
# min_date = leave_df['Leave_Start'].min()
# max_date = leave_df['Leave_Start'].max()
# start_date, end_date = st.sidebar.date_input("Select Date Range:", [min_date, max_date], min_value=min_date, max_value=max_date)

# # Apply Filters
# filtered_df = leave_df[
#     ((leave_df['Employee'] == selected_employee) | (selected_employee == "All")) &
#     (leave_df['Leave_Type'].isin(selected_leave_types)) &
#     (leave_df['Leave_Start'].between(pd.to_datetime(start_date), pd.to_datetime(end_date)))
# ]

# # Display filtered data table
# st.subheader("📋 Filtered Leave Data")
# st.dataframe(filtered_df)

# # ----- Analytics Section -----
# if not filtered_df.empty:
    
#     # --- Average Leave Duration ---
#     avg_duration = round(filtered_df['Leave_Duration'].mean(), 2)
#     st.subheader("📅 Average Leave Duration")
#     st.metric(label="Average Leave Duration (Days)", value=avg_duration)

#     # --- Leave Type Distribution (Pie Chart) ---
#     st.subheader("🥧 Leave Type Distribution")
#     leave_type_counts = filtered_df['Leave_Type'].value_counts()
#     # Add explode effect to separate slices
#     explode = [0.1] * len(leave_type_counts)  # Slightly explode all slices for better visualization


#     fig, ax = plt.subplots(figsize=(5, 5))
#     ax.pie(leave_type_counts, labels=leave_type_counts.index, autopct='%1.1f%%', colors=sns.color_palette("Set2"), startangle=140, explode=explode)
#     ax.set_title("Leave Type Distribution")
#     st.pyplot(fig)

#     # --- Leave Trend by Month ---
#     leave_trend = filtered_df.groupby(['Month_Name', 'Leave_Type']).size().reset_index(name='Leave_Count')

#     if not leave_trend.empty:
#         leave_pivot = leave_trend.pivot(index='Month_Name', columns='Leave_Type', values='Leave_Count').fillna(0)

#         st.subheader("📈 Leave Trend by Month & Type")
#         fig, ax = plt.subplots(figsize=(10, 5))
#         leave_pivot.plot(kind='bar', stacked=True, colormap='Set2', ax=ax)

#         ax.set_xlabel("Month")
#         ax.set_ylabel("Number of Leave Applications")
#         ax.set_title(f"Leave Trend by Month & Type ({selected_employee})")
#         ax.legend(title="Leave Type")
#         ax.grid(axis='y', linestyle='--', alpha=0.7)
#         st.pyplot(fig)

#     # --- Heatmap for Monthly Leave Trends ---
#     st.subheader("🔥 Monthly Leave Trend Heatmap")
#     heatmap_data = filtered_df.groupby(['Month_Name', 'Leave_Type'])['Leave_Duration'].sum().unstack().fillna(0)

#     fig, ax = plt.subplots(figsize=(10, 5))
#     sns.heatmap(heatmap_data, cmap="Blues", annot=True, fmt=".1f", linewidths=0.5, ax=ax)
#     ax.set_title("Total Leave Days by Month & Type")
#     st.pyplot(fig)

# else:
#     st.warning("⚠️ No data available for the selected filters.")











# GREAT AND AMAZING LEAVE APPLICATION CODE THAT FIX THE WEEKDAYS 2.



import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Sample leave application data
datafile = "employee_leave_2025.xlsx"

# Check if the file exists
if not os.path.exists(datafile):
    st.error("Error: The file 'employee_leave_2025.xlsx' was not found.")
    st.stop()  # Stop execution if file is missing

# Convert to DataFrame
leave_df = pd.read_excel(datafile)

# Ensure required columns exist
required_columns = {'Employee_Name', 'Start_Date', 'End_Date', 'Reason'}
if not required_columns.issubset(leave_df.columns):
    st.error(f"Error: Missing required columns: {required_columns - set(leave_df.columns)}")
    st.stop()

# Convert dates to datetime
leave_df['Start_Date'] = pd.to_datetime(leave_df['Start_Date'])
leave_df['End_Date'] = pd.to_datetime(leave_df['End_Date'])

# Calculate working days (Monday-Friday)
leave_df['Working Days'] = leave_df.apply(lambda row: np.busday_count(row['Start_Date'].date(), row['End_Date'].date()), axis=1)


# Calculate Leave Duration in Days
leave_df['Leave_Duration'] = leave_df.apply(
    lambda row: np.busday_count(row['Start_Date'].date(), row['End_Date'].date()) + 1,
    axis=1)

# Extract Month Name from Start_Date
leave_df['Month_Name'] = leave_df['Start_Date'].dt.strftime('%B')

# Define month order for sorting
month_order = ['January', 'February', 'March', 'April', 'May', 'June', 
               'July', 'August', 'September', 'October', 'November', 'December']

# Convert Month_Name column to categorical
leave_df['Month_Name'] = pd.Categorical(leave_df['Month_Name'], categories=month_order, ordered=True)

# Sidebar Filters
selected_employee = st.sidebar.selectbox("Select Employee", ["All"] + sorted(leave_df["Employee_Name"].dropna().unique()))
selected_leave_types = st.sidebar.multiselect("Select Leave Type", leave_df["Reason"].dropna().unique(), default=leave_df["Reason"].dropna().unique())

# Apply filters
filtered_df = leave_df[
    ((leave_df['Employee_Name'] == selected_employee) | (selected_employee == "All")) &
    (leave_df['Reason'].isin(selected_leave_types))
]

# Streamlit app layout
st.title("📊 Customer Service Leave Trend Analysis by Month & Type")

st.write("""
This dashboard visualizes the number of leave applications per month, categorized by leave type.
It helps Head of OPeration, Cs Supervisors mostly to track trends and manage leave distribution and planning.
""")

# Display filtered data table
with st.expander("📋 View Filtered Leave Data"):
    st.dataframe(filtered_df)

# Calculate key leave metrics
total_leaves = len(filtered_df)
total_leave_days = filtered_df['Leave_Duration'].sum() if total_leaves > 0 else 0
average_leave_days = round(filtered_df['Leave_Duration'].mean(), 2) if total_leaves > 0 else 0

# Layout for displaying metrics
col1, col2, col3 = st.columns(3)
col1.metric(label="📊 Total Leave Applications", value=total_leaves)
col2.metric(label="📅 Total Leave Days", value=total_leave_days)
col3.metric(label="📊 Avg Leave Duration (Days)", value=average_leave_days)

st.subheader("Leave summary")
leave_status_summary = filtered_df['Reason'].value_counts()


total_leave = {
    "Annual": leave_status_summary.get("Annual Leave", 0),
    "Sick": leave_status_summary.get("Sick Leave", 0),
     "Study": leave_status_summary.get("Study Leave", 0),
    "Casual": leave_status_summary.get("Casual Leave", 0),
    "Maternity": leave_status_summary.get("Maternity Leave", 0),
    "Paternity": leave_status_summary.get("Paternity Leave", 0)
}

st.subheader("Leave Summary by Type")
st.write(f"**Total Annual Leave**: {total_leave['Annual']}")
st.write(f"**Total Sick Leave**: {total_leave['Sick']}") 
st.write(f"**Total Study Leave**: {total_leave['Study']}") 
st.write(f"**Total Casual Leave**: {total_leave['Casual']}")
st.write(f"**Total Maternity Leave**: {total_leave['Maternity']}")
st.write(f"**Total Paternity Leave**: {total_leave['Paternity']}")




# --- Leave Trend by Month ---
leave_trend = filtered_df.groupby(['Month_Name', 'Reason']).size().reset_index(name='Leave_Count')


if not leave_trend.empty:
    leave_pivot = leave_trend.pivot(index='Month_Name', columns='Reason', values='Leave_Count').fillna(0)

    st.subheader("📈 Leave Trend by Month & Type")
    fig, ax = plt.subplots(figsize=(10, 5))
    leave_pivot.plot(kind='bar', stacked=True, colormap='Set2', ax=ax)

    ax.set_xlabel("Month")
    ax.set_ylabel("Number of Leave Applications")
    ax.set_title(f"Leave Trend by Month & Type ({selected_employee})")
    ax.legend(title="Reason")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    st.pyplot(fig)

    # --- Heatmap for Monthly Leave Trends ---
    st.subheader("🔥 Monthly Leave Trend Heatmap")
    heatmap_data = filtered_df.groupby(['Month_Name', 'Reason'])['Leave_Duration'].sum().unstack().fillna(0)

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(heatmap_data, cmap="Blues", annot=True, fmt=".1f", linewidths=0.5, ax=ax)
    ax.set_title("Total Leave Days by Month & Type")
    st.pyplot(fig)
else:
    st.warning("⚠️ No data available for the selected filters.")



# Define the correct month order
month_order = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December"
]

# Ensure 'Month_Name' is a categorical variable with the correct order
leave_trend = filtered_df.groupby(['Employee_Name', 'Month_Name']).size().reset_index(name='Leave_Taken_Each_Month_By_Staff')

# Group by Month to get total leave requests per month
monthly_leave = leave_trend.groupby('Month_Name')['Leave_Taken_Each_Month_By_Staff'].sum().reset_index()

# Convert 'Month_Name' to categorical and sort accordingly
monthly_leave['Month_Name'] = pd.Categorical(monthly_leave['Month_Name'], categories=month_order, ordered=True)
monthly_leave = monthly_leave.sort_values('Month_Name')

# Display results
st.subheader('Determine the month with the highest number of leave requests')
st.table(monthly_leave)



# ---- Leave Type Breakdown (Pie Chart) ----
st.subheader("📊 Leave Type Breakdown")
if not leave_status_summary.empty:

    fig, ax = plt.subplots()
    colors = sns.color_palette("Set2", len(leave_status_summary))
    pd.Series(leave_status_summary).plot(kind='pie', autopct='%1.1f%%', startangle=90, colors=colors, ax=ax)
    ax.set_ylabel("")
    ax.set_title("Percentage of Leave Types")
    st.pyplot(fig)

else:
    st.warning("⚠️ No leave data available for pie chart.")    

# ---- Leave Distribution by Employee ----
if not filtered_df.empty:
    leave_distribution = filtered_df.groupby(['Reason', 'Employee_Name'])['Leave_Duration'].sum().reset_index().sort_values('Leave_Duration')
    
    st.subheader("📊 Leave Distribution by Type & Employee")
    # st.write(leave_distribution)

    with st.expander("📋 View Leave Breakdown by Employee"):
        st.write(leave_distribution)
else:
    st.warning("⚠️ No data available for leave distribution.")


leave_counts = leave_df['Employee_Name'].value_counts()

# Get the employee with the highest leave count
top_employee = leave_counts.idxmax()  # Employee with most leaves
top_leave_count = leave_counts.max()  # Number of leaves taken

st.subheader("🏆 Employee with Most Leave")
st.write(f"**{top_employee}** has taken the most leave, with **{top_leave_count}** applications.")

# Display full leave count table
with st.expander("📋 View Leave Counts for All Employees"):
    st.write(leave_counts)


# Count the number of leaves taken by each employee
leave_counts = leave_df['Employee_Name'].value_counts()

# Get the employee with the least leave count
least_employee = leave_counts.idxmin()  # Employee with least leaves
least_leave_count = leave_counts.min()  # Minimum number of leaves taken

st.subheader("🥇 Employee with Least Leave")
st.write(f"**{least_employee}** has taken the least leave, with **{least_leave_count}** application(s).")

# Display full leave count table
with st.expander("📋 View Leave Counts for All Employees"):
    st.write(leave_counts)    


 # Calculate total leave days for each employee
leave_df['Leave_Duration'] = (leave_df['End_Date'] - leave_df['Start_Date']).dt.days + 1
leave_summary = leave_df.groupby('Employee_Name')['Leave_Duration'].sum().reset_index()

# Get the top 5 employees with the most leave taken
top_5_most_leave = leave_summary.nlargest(5, 'Leave_Duration')

# Get the bottom 5 employees with the least leave taken
bottom_5_least_leave = leave_summary.nsmallest(5, 'Leave_Duration')


st.markdown(
    "<hr style='border: 2px solid green;'>", unsafe_allow_html=True
)


# Display in Streamlit
st.subheader("🏆 Employees with Most & Least Leave Taken")

col1, col2 = st.columns(2)

with col1:
    st.write("**Top 5 Employees with Most Leave Taken**")
    st.dataframe(top_5_most_leave)

with col2:
    st.write("**Bottom 5 Employees with Least Leave Taken**")
    st.dataframe(bottom_5_least_leave)
   












# # GREAT AND AMAZING LEAVE APPLICATION  CODE 1.

# import pandas as pd
# import streamlit as st
# import matplotlib.pyplot as plt
# import seaborn as sns
# import os

# # Sample leave application data
# datafile = "employee_leave_2025.xlsx"

# # Check if the file exists
# if not os.path.exists(datafile):
#     st.error("Error: The file 'employee_leave_2025.xlsx' was not found.")
#     st.stop()  # Stop execution if file is missing

# # Convert to DataFrame
# leave_df = pd.read_excel(datafile)

# # Ensure required columns exist
# required_columns = {'Employee_Name', 'Start_Date', 'End_Date', 'Reason'}
# if not required_columns.issubset(leave_df.columns):
#     st.error(f"Error: Missing required columns: {required_columns - set(leave_df.columns)}")
#     st.stop()

# # Convert date columns to datetime format
# leave_df['Start_Date'] = pd.to_datetime(leave_df['Start_Date'], errors='coerce')
# leave_df['End_Date'] = pd.to_datetime(leave_df['End_Date'], errors='coerce')

# # Drop rows with invalid dates
# leave_df.dropna(subset=['Start_Date', 'End_Date'], inplace=True)

# # Calculate Leave Duration in Days
# leave_df['Leave_Duration'] = (leave_df['End_Date'] - leave_df['Start_Date']).dt.days+1

# # Extract Month Name from Start_Date
# leave_df['Month_Name'] = leave_df['Start_Date'].dt.strftime('%B')

# # Define month order for sorting
# month_order = ['January', 'February', 'March', 'April', 'May', 'June', 
#                'July', 'August', 'September', 'October', 'November', 'December']

# # Convert Month_Name column to categorical
# leave_df['Month_Name'] = pd.Categorical(leave_df['Month_Name'], categories=month_order, ordered=True)

# # Sidebar Filters
# selected_employee = st.sidebar.selectbox("Select Employee", ["All"] + sorted(leave_df["Employee_Name"].dropna().unique()))
# selected_leave_types = st.sidebar.multiselect("Select Leave Type", leave_df["Reason"].dropna().unique(), default=leave_df["Reason"].dropna().unique())

# # Apply filters
# filtered_df = leave_df[
#     ((leave_df['Employee_Name'] == selected_employee) | (selected_employee == "All")) &
#     (leave_df['Reason'].isin(selected_leave_types))
# ]

# # Streamlit app layout
# st.title("📊 Customer Service Leave Trend Analysis by Month & Type")

# st.write("""
# This dashboard visualizes the number of leave applications per month, categorized by leave type.
# It helps HR track trends and manage leave planning.
# """)

# # Display filtered data table
# # with st.expander("📋 View Filtered Leave Data"):
# #     st.dataframe(filtered_df)

# # Calculate key leave metrics
# total_leaves = len(filtered_df)
# total_leave_days = filtered_df['Leave_Duration'].sum() if total_leaves > 0 else 0
# average_leave_days = round(filtered_df['Leave_Duration'].mean(), 2) if total_leaves > 0 else 0

# # Layout for displaying metrics
# col1, col2, col3 = st.columns(3)
# col1.metric(label="📊 Total Leave Applications", value=total_leaves)
# col2.metric(label="📅 Total Leave Days", value=total_leave_days)
# col3.metric(label="📊 Avg Leave Duration (Days)", value=average_leave_days)

# st.subheader("Leave summary")
# leave_status_summary = filtered_df['Reason'].value_counts()


# total_leave = {
#     "Annual": leave_status_summary.get("Annual Leave", 0),
#     "Sick": leave_status_summary.get("Sick Leave", 0),
#     "Casual": leave_status_summary.get("Casual Leave", 0),
#     "Maternity": leave_status_summary.get("Maternity Leave", 0),
#     "Paternity": leave_status_summary.get("Paternity Leave", 0)
# }

# st.subheader("Leave Summary by Type")
# st.write(f"**Total Annual Leave**: {total_leave['Annual']}")
# st.write(f"**Total Sick Leave**: {total_leave['Sick']}") 
# st.write(f"**Total Casual Leave**: {total_leave['Casual']}")
# st.write(f"**Total Maternity Leave**: {total_leave['Maternity']}")
# st.write(f"**Total Paternity Leave**: {total_leave['Paternity']}")




# # --- Leave Trend by Month ---
# leave_trend = filtered_df.groupby(['Month_Name', 'Reason']).size().reset_index(name='Leave_Count')


# if not leave_trend.empty:
#     leave_pivot = leave_trend.pivot(index='Month_Name', columns='Reason', values='Leave_Count').fillna(0)

#     st.subheader("📈 Leave Trend by Month & Type")
#     fig, ax = plt.subplots(figsize=(10, 5))
#     leave_pivot.plot(kind='bar', stacked=True, colormap='Set2', ax=ax)

#     ax.set_xlabel("Month")
#     ax.set_ylabel("Number of Leave Applications")
#     ax.set_title(f"Leave Trend by Month & Type ({selected_employee})")
#     ax.legend(title="Reason")
#     ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
#     ax.grid(axis='y', linestyle='--', alpha=0.7)
#     st.pyplot(fig)

#     # --- Heatmap for Monthly Leave Trends ---
#     st.subheader("🔥 Monthly Leave Trend Heatmap")
#     heatmap_data = filtered_df.groupby(['Month_Name', 'Reason'])['Leave_Duration'].sum().unstack().fillna(0)

#     fig, ax = plt.subplots(figsize=(10, 5))
#     sns.heatmap(heatmap_data, cmap="Blues", annot=True, fmt=".1f", linewidths=0.5, ax=ax)
#     ax.set_title("Total Leave Days by Month & Type")
#     st.pyplot(fig)
# else:
#     st.warning("⚠️ No data available for the selected filters.")

# # # ---- Monthly Leave Distribution Chart ----
# # st.subheader("📊 Monthly Leave Applications")
# # if not filtered_df.empty:
# #     fig, ax = plt.subplots(figsize=(10, 5))
# #     sns.countplot(data=filtered_df, x='Month_Name', order=month_order, hue='Reason', palette='coolwarm', ax=ax)

# #     ax.set_xlabel("Month")
# #     ax.set_ylabel("Number of Leave Applications")
# #     ax.set_title("Leave Applications by Month")
# #     ax.legend(title="Leave Type")
# #     ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
# #     st.pyplot(fig)
# # else:
# #     st.warning("⚠️ No data available to plot.")




# # Group by Employee and Month to count leave occurrences
# leave_trend = filtered_df.groupby(['Employee_Name', 'Month_Name']).size().reset_index(name='Total_Leave')

# # Group by Month to get total leave requests per month and sort in descending order
# monthly_leave = leave_trend.groupby('Month_Name')['Total_Leave'].sum().reset_index().sort_values(by='Total_Leave', ascending=False)

# # Display results
# st.subheader('Determine the month with the highest number of leave requests')
# st.table(monthly_leave)



# # ---- Leave Type Breakdown (Pie Chart) ----
# st.subheader("📊 Leave Type Breakdown")
# if not leave_status_summary.empty:

#     fig, ax = plt.subplots()
#     colors = sns.color_palette("Set2", len(leave_status_summary))
#     pd.Series(leave_status_summary).plot(kind='pie', autopct='%1.1f%%', startangle=90, colors=colors, ax=ax)
#     ax.set_ylabel("")
#     ax.set_title("Percentage of Leave Types")
#     st.pyplot(fig)

# else:
#     st.warning("⚠️ No leave data available for pie chart.")    

# # ---- Leave Distribution by Employee ----
# if not filtered_df.empty:
#     leave_distribution = filtered_df.groupby(['Reason', 'Employee_Name'])['Leave_Duration'].sum().reset_index().sort_values('Leave_Duration')
    
#     st.subheader("📊 Leave Distribution by Type & Employee")
#     # st.write(leave_distribution)

#     with st.expander("📋 View Leave Breakdown by Employee"):
#         st.write(leave_distribution)
# else:
#     st.warning("⚠️ No data available for leave distribution.")


# leave_counts = leave_df['Employee_Name'].value_counts()

# # Get the employee with the highest leave count
# top_employee = leave_counts.idxmax()  # Employee with most leaves
# top_leave_count = leave_counts.max()  # Number of leaves taken

# st.subheader("🏆 Employee with Most Leave")
# st.write(f"**{top_employee}** has taken the most leave, with **{top_leave_count}** applications.")

# # Display full leave count table
# with st.expander("📋 View Leave Counts for All Employees"):
#     st.write(leave_counts)


# # Count the number of leaves taken by each employee
# leave_counts = leave_df['Employee_Name'].value_counts()

# # Get the employee with the least leave count
# least_employee = leave_counts.idxmin()  # Employee with least leaves
# least_leave_count = leave_counts.min()  # Minimum number of leaves taken

# st.subheader("🥇 Employee with Least Leave")
# st.write(f"**{least_employee}** has taken the least leave, with **{least_leave_count}** application(s).")

# # Display full leave count table
# with st.expander("📋 View Leave Counts for All Employees"):
#     st.write(leave_counts)    


#  # Calculate total leave days for each employee
# leave_df['Leave_Duration'] = (leave_df['End_Date'] - leave_df['Start_Date']).dt.days + 1
# leave_summary = leave_df.groupby('Employee_Name')['Leave_Duration'].sum().reset_index()

# # Get the top 5 employees with the most leave taken
# top_5_most_leave = leave_summary.nlargest(5, 'Leave_Duration')

# # Get the bottom 5 employees with the least leave taken
# bottom_5_least_leave = leave_summary.nsmallest(5, 'Leave_Duration')

# # Display in Streamlit
# st.subheader("🏆 Employees with Most & Least Leave Taken")

# col1, col2 = st.columns(2)

# with col1:
#     st.write("**Top 5 Employees with Most Leave Taken**")
#     st.dataframe(top_5_most_leave)

# with col2:
#     st.write("**Bottom 5 Employees with Least Leave Taken**")
#     st.dataframe(bottom_5_least_leave)
   









# # Count the number of leaves taken by each employee
# leave_counts = leave_df['Employee_Name'].value_counts()

# # Get the employee with the highest leave count
# top_employee = leave_counts.idxmax()  # Employee with most leaves
# top_leave_count = leave_counts.max()  # Number of leaves taken

# st.subheader("🏆 Employee with Most Leave")
# st.write(f"**{top_employee}** has taken the most leave, with **{top_leave_count}** applications.")

# # Display full leave count table
# with st.expander("📋 View Leave Counts for All Employees"):
#     st.write(leave_counts)


# # Count the number of leaves taken by each employee
# leave_counts = leave_df['Employee'].value_counts()

# # Get the employee with the least leave count
# least_employee = leave_counts.idxmin()  # Employee with least leaves
# least_leave_count = leave_counts.min()  # Minimum number of leaves taken

# st.subheader("🥇 Employee with Least Leave")
# st.write(f"**{least_employee}** has taken the least leave, with **{least_leave_count}** application(s).")

# # Display full leave count table
# with st.expander("📋 View Leave Counts for All Employees"):
#     st.write(leave_counts)



# # Assume each employee has an annual leave entitlement of 20 days
# LEAVE_ENTITLEMENT = 20

# # Calculate leave days taken per employee
# leave_df['Leave_Days'] = (leave_df['Leave_End'] - leave_df['Leave_Start']).dt.days + 1
# leave_taken = leave_df.groupby('Employee')['Leave_Days'].sum()

# # Calculate remaining leave per employee
# leave_remaining = LEAVE_ENTITLEMENT - leave_taken

# # Convert negative values (if any) to 0 (means employee exhausted all leave)
# leave_remaining = leave_remaining.clip(lower=0)

# # Display in Streamlit
# with st.expander("📋 View Detailed Leave Data"):
#  st.write(leave_remaining)



# # Calculate leave days taken for each leave record
# leave_df['Leave_Days'] = (leave_df['Leave_End'] - leave_df['Leave_Start']).dt.days + 1

# # Group by Leave_Type and sum the total leave days
# leave_by_type = leave_df.groupby('Leave_Type')['Leave_Days'].sum().reset_index()

# # Display in Streamlit
# st.subheader("📊 Total Leave Taken by Type")
# st.write(leave_by_type)

# # Optional: Display in an expander
# with st.expander("📋 View Detailed Leave Data"):
#     st.write(leave_df)


# # Calculate total leave days taken per employee for each leave type
# leave_df['Leave_Days'] = (leave_df['Leave_End'] - leave_df['Leave_Start']).dt.days + 1

# # Group by Leave_Type and Employee
# leave_distribution = leave_df.groupby(['Leave_Type', 'Employee'])['Leave_Days'].sum().reset_index().sort_values('Leave_Days')

# # Display in Streamlit
# st.subheader("📊 Leave Distribution by Type & Employee")
# st.write(leave_distribution)

# # Optional: Use an expander
# with st.expander("📋 View Leave Breakdown by Employee"):
#     st.write(leave_distribution)


# Assuming each employee has a fixed leave entitlement (e.g., 20 days per year)
# leave_entitlement = 20  

# # Calculate total leave taken by each employee
# leave_taken_summary = leave_df.groupby('Employee').agg({'Leave_Start': 'count'}).reset_index()
# leave_taken_summary.rename(columns={'Leave_Start': 'Total_Leave_Taken'}, inplace=True)

# # Calculate remaining leave
# leave_taken_summary['Remaining_Leave'] = leave_entitlement - leave_taken_summary['Total_Leave_Taken']

# # Identify employees who have taken all their leave (Remaining_Leave == 0)
# employees_exhausted_leave = leave_taken_summary[leave_taken_summary['Remaining_Leave'] <= 0]

# # Display results in Streamlit
# st.subheader("✅ Employees Who Have Taken All Their Leave")
# with st.expander("📋 View List of Employees Who Exhausted Their Leave"):
#     st.dataframe(employees_exhausted_leave)





# # Count the number of unique employees taking leave each month
# employees_per_month = leave_df.groupby('Month_Name')['Employee'].nunique().reset_index()

# # Rename columns for better clarity
# employees_per_month.columns = ['Month_Name', 'Employees_on_Leave']

# # Sort by month order
# employees_per_month['Month_Name'] = pd.Categorical(employees_per_month['Month_Name'], 
#                                                    categories=month_order, ordered=True)
# employees_per_month = employees_per_month.sort_values('Month_Name')

# # Display results in Streamlit
# st.subheader("📅 Number of Employees Taking Leave Each Month")
# with st.expander("📋 View Monthly Leave Summary"):
#     st.dataframe(employees_per_month)

# # Plotting the data
# fig, ax = plt.subplots(figsize=(10, 5))
# sns.barplot(x='Month_Name', y='Employees_on_Leave', data=employees_per_month, palette='Set2', ax=ax)

# # Customize plot
# ax.set_xlabel("Month")
# ax.set_ylabel("Number of Employees")
# ax.set_title("Employees Taking Leave Each Month")
# ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
# ax.grid(axis='y', linestyle='--', alpha=0.7)

# # Display plot in Streamlit
# st.pyplot(fig)




# # Calculate total leave days for each employee
# leave_df['Leave_Days'] = (leave_df['Leave_End'] - leave_df['Leave_Start']).dt.days + 1
# leave_summary = leave_df.groupby('Employee')['Leave_Days'].sum().reset_index()

# # Get the top 5 employees with the most leave taken
# top_5_most_leave = leave_summary.nlargest(5, 'Leave_Days')

# # Get the bottom 5 employees with the least leave taken
# bottom_5_least_leave = leave_summary.nsmallest(5, 'Leave_Days')

# # Display in Streamlit
# st.subheader("🏆 Employees with Most & Least Leave Taken")

# col1, col2 = st.columns(2)

# with col1:
#     st.write("**Top 5 Employees with Most Leave Taken**")
#     st.dataframe(top_5_most_leave)

# with col2:
#     st.write("**Bottom 5 Employees with Least Leave Taken**")
#     st.dataframe(bottom_5_least_leave)




# # Calculate Leave Days
# leave_df['Leave_Days'] = (leave_df['Leave_End'] - leave_df['Leave_Start']).dt.days + 1
# leave_summary = leave_df.groupby('Employee')['Leave_Days'].sum().reset_index()

# # Top 5 Employees with Most Leave Taken
# top_5_most_leave = leave_summary.nlargest(5, 'Leave_Days')

# # Bottom 5 Employees with Least Leave Taken
# bottom_5_least_leave = leave_summary.nsmallest(5, 'Leave_Days')

# # Streamlit Layout
# st.subheader("🏆 Employees with Most & Least Leave Taken")

# # Plot Top 5 Employees with Most Leave (Stacked)
# st.write("### 🏅 Top 5 Employees with Most Leave Taken")
# fig, ax = plt.subplots(figsize=(8, 5))
# sns.barplot(y=top_5_most_leave['Employee'], x=top_5_most_leave['Leave_Days'], palette='Blues_r', ax=ax)
# ax.set_xlabel("Total Leave Days")
# ax.set_ylabel("Employee")
# ax.set_title("Most Leave Taken")
# st.pyplot(fig)

# # Plot Bottom 5 Employees with Least Leave (Stacked)
# st.write("### 📉 Bottom 5 Employees with Least Leave Taken")
# fig, ax = plt.subplots(figsize=(8, 5))
# sns.barplot(y=bottom_5_least_leave['Employee'], x=bottom_5_least_leave['Leave_Days'], palette='Reds_r', ax=ax)
# ax.set_xlabel("Total Leave Days")
# ax.set_ylabel("Employee")
# ax.set_title("Least Leave Taken")
# st.pyplot(fig)






# # Leave Type Breakdown
# with st.expander("📊 View Leave Type Breakdown"):
#  st.subheader("📊 Leave Type Breakdown")
#  col4, col5, col6, col7 = st.columns(4)
#  col4.metric(label="Annual Leave", value=total_leave["Annual"])
#  col5.metric(label="Sick Leave", value=total_leave["Sick"])
#  col6.metric(label="Casual Leave", value=total_leave["Casual"])
#  col7.metric(label="Maternity Leave", value=total_leave["Maternity"])

# # Group by Month_Name and Leave_Type
# # Group by Month_Name and Leave_Type
# leave_trend = filtered_df.groupby(['Month_Name', 'Leave_Type']).size().reset_index(name='Leave_Count')

# # Pivot for Stacked Bar Chart
# leave_pivot = leave_trend.pivot(index='Month_Name', columns='Leave_Type', values='Leave_Count').fillna(0)

# # Convert all values to numeric (avoiding non-numeric issues)
# # leave_pivot = leave_pivot.apply(pd.to_numeric, errors='coerce')

# # Reset index for easier viewing
# leave_pivot_reset = leave_pivot.reset_index()

# # Using expander to hide pivot table
# with st.expander("📋 View Leave Pivot Table"):
#     st.write(leave_pivot_reset)

# # Ensure the pivot table has data before plotting
# if leave_pivot.empty:
#     st.warning("⚠ No leave data available for the selected filters.")
# else:
#     # Display leave trend chart
#     st.subheader("📈 Leave Trend by Month & Type")
#     fig, ax = plt.subplots(figsize=(10, 5))
#     leave_pivot.plot(kind='bar', stacked=True, colormap='Set2', ax=ax)

#     ax.set_xlabel("Month")
#     ax.set_ylabel("Number of Leave Applications")
#     ax.set_title("Leave Trend by Month & Type")
#     ax.legend(title="Leave Type")
#     ax.grid(axis='y', linestyle='--', alpha=0.7)

#     st.pyplot(fig)











# LEAVE APPLICATION 1

# import pandas as pd
# import streamlit as st
# import matplotlib.pyplot as plt
# import seaborn as sns

# # Sample leave application data
# data = {
#     'Employee': ['John', 'Sarah', 'Mike', 'Anna', 'David', 'Emily', 'Tom', 'Sophia', 'Chris', 'Emma'],
#     'Leave_Start': ['2024-01-10', '2024-02-15', '2024-02-20', '2024-03-05', '2024-03-18',
#                     '2024-04-25', '2024-04-30', '2024-05-10', '2024-02-25', '2024-04-12'],
#     'Leave_End': ['2024-01-17', '2024-02-18', '2024-02-22', '2024-03-07', '2024-03-22',
#                   '2024-04-28', '2024-05-02', '2024-05-12', '2024-02-28', '2024-04-15'],
#     'Leave_Type': ['Annual', 'Sick', 'Annual', 'Maternity', 'Casual', 'Sick', 'Annual', 'Casual', 'Sick', 'Annual']
# }

# # Convert to DataFrame
# leave_df = pd.DataFrame(data)

# # Convert date columns to datetime format
# leave_df['Leave_Start'] = pd.to_datetime(leave_df['Leave_Start'])
# leave_df['Leave_End'] = pd.to_datetime(leave_df['Leave_End'])

# # Calculate Leave Duration in Days
# leave_df['Leave_Days'] = (leave_df['Leave_End'] - leave_df['Leave_Start']).dt.days + 1

# # Extract Month Name from Leave_Start date
# leave_df['Month_Name'] = leave_df['Leave_Start'].dt.strftime('%B')

# # Define month order for sorting
# month_order = ['January', 'February', 'March', 'April', 'May', 'June', 
#                'July', 'August', 'September', 'October', 'November', 'December']

# # Convert Month_Name column to categorical
# leave_df['Month_Name'] = pd.Categorical(leave_df['Month_Name'], categories=month_order, ordered=True)

# # Filters
# selected_employee = st.sidebar.selectbox("Select Employee", ["All"] + list(leave_df["Employee"].unique()))
# selected_leave_types = st.sidebar.multiselect("Select Leave Type", leave_df["Leave_Type"].unique(), default=leave_df["Leave_Type"].unique())

# # Apply filters
# filtered_df = leave_df[
#     ((leave_df['Employee'] == selected_employee) | (selected_employee == "All")) &
#     (leave_df['Leave_Type'].isin(selected_leave_types))
# ]

# # Leave status summary
# leave_status_summary = filtered_df['Leave_Type'].value_counts()


# # Total leave breakdown
# total_leave = {
#     "Annual": leave_status_summary.get("Annual", 0),
#     "Sick": leave_status_summary.get("Sick", 0),
#     "Casual": leave_status_summary.get("Casual", 0),
#     "Maternity": leave_status_summary.get("Maternity", 0)  # Fixed capitalization
# }

# # Streamlit app layout
# st.title("📊 Leave Trend Analysis by Month & Type")

# st.write("""
# This dashboard visualizes the number of leave applications per month, categorized by leave type.
# It helps HR track trends and manage leave planning.
# """)


# # Display filtered data table
# with st.expander("📋 View Filtered Leave Data"):
#     st.dataframe(filtered_df)


# # Count the number of leaves taken by each employee
# leave_counts = leave_df['Employee'].value_counts()

# # Get the employee with the highest leave count
# top_employee = leave_counts.idxmax()  # Employee with most leaves
# top_leave_count = leave_counts.max()  # Number of leaves taken

# st.subheader("🏆 Employee with Most Leave")
# st.write(f"**{top_employee}** has taken the most leave, with **{top_leave_count}** applications.")

# # Display full leave count table
# with st.expander("📋 View Leave Counts for All Employees"):
#     st.write(leave_counts)


# # Count the number of leaves taken by each employee
# leave_counts = leave_df['Employee'].value_counts()

# # Get the employee with the least leave count
# least_employee = leave_counts.idxmin()  # Employee with least leaves
# least_leave_count = leave_counts.min()  # Minimum number of leaves taken

# st.subheader("🥇 Employee with Least Leave")
# st.write(f"**{least_employee}** has taken the least leave, with **{least_leave_count}** application(s).")

# # Display full leave count table
# with st.expander("📋 View Leave Counts for All Employees"):
#     st.write(leave_counts)



# # Assume each employee has an annual leave entitlement of 20 days
# LEAVE_ENTITLEMENT = 20

# # Calculate leave days taken per employee
# leave_df['Leave_Days'] = (leave_df['Leave_End'] - leave_df['Leave_Start']).dt.days + 1
# leave_taken = leave_df.groupby('Employee')['Leave_Days'].sum()

# # Calculate remaining leave per employee
# leave_remaining = LEAVE_ENTITLEMENT - leave_taken

# # Convert negative values (if any) to 0 (means employee exhausted all leave)
# leave_remaining = leave_remaining.clip(lower=0)

# # Display in Streamlit
# with st.expander("📋 View Detailed Leave Data"):
#  st.write(leave_remaining)



# # Calculate leave days taken for each leave record
# leave_df['Leave_Days'] = (leave_df['Leave_End'] - leave_df['Leave_Start']).dt.days + 1

# # Group by Leave_Type and sum the total leave days
# leave_by_type = leave_df.groupby('Leave_Type')['Leave_Days'].sum().reset_index()

# # Display in Streamlit
# st.subheader("📊 Total Leave Taken by Type")
# st.write(leave_by_type)

# # Optional: Display in an expander
# with st.expander("📋 View Detailed Leave Data"):
#     st.write(leave_df)


# # Calculate total leave days taken per employee for each leave type
# leave_df['Leave_Days'] = (leave_df['Leave_End'] - leave_df['Leave_Start']).dt.days + 1

# # Group by Leave_Type and Employee
# leave_distribution = leave_df.groupby(['Leave_Type', 'Employee'])['Leave_Days'].sum().reset_index().sort_values('Leave_Days')

# # Display in Streamlit
# st.subheader("📊 Leave Distribution by Type & Employee")
# st.write(leave_distribution)

# # Optional: Use an expander
# with st.expander("📋 View Leave Breakdown by Employee"):
#     st.write(leave_distribution)


# # Assuming each employee has a fixed leave entitlement (e.g., 20 days per year)
# leave_entitlement = 20  

# # Calculate total leave taken by each employee
# leave_taken_summary = leave_df.groupby('Employee').agg({'Leave_Start': 'count'}).reset_index()
# leave_taken_summary.rename(columns={'Leave_Start': 'Total_Leave_Taken'}, inplace=True)

# # Calculate remaining leave
# leave_taken_summary['Remaining_Leave'] = leave_entitlement - leave_taken_summary['Total_Leave_Taken']

# # Identify employees who have taken all their leave (Remaining_Leave == 0)
# employees_exhausted_leave = leave_taken_summary[leave_taken_summary['Remaining_Leave'] <= 0]

# # Display results in Streamlit
# st.subheader("✅ Employees Who Have Taken All Their Leave")
# with st.expander("📋 View List of Employees Who Exhausted Their Leave"):
#     st.dataframe(employees_exhausted_leave)





# # Count the number of unique employees taking leave each month
# employees_per_month = leave_df.groupby('Month_Name')['Employee'].nunique().reset_index()

# # Rename columns for better clarity
# employees_per_month.columns = ['Month_Name', 'Employees_on_Leave']

# # Sort by month order
# employees_per_month['Month_Name'] = pd.Categorical(employees_per_month['Month_Name'], 
#                                                    categories=month_order, ordered=True)
# employees_per_month = employees_per_month.sort_values('Month_Name')

# # Display results in Streamlit
# st.subheader("📅 Number of Employees Taking Leave Each Month")
# with st.expander("📋 View Monthly Leave Summary"):
#     st.dataframe(employees_per_month)

# # Plotting the data
# fig, ax = plt.subplots(figsize=(10, 5))
# sns.barplot(x='Month_Name', y='Employees_on_Leave', data=employees_per_month, palette='Set2', ax=ax)

# # Customize plot
# ax.set_xlabel("Month")
# ax.set_ylabel("Number of Employees")
# ax.set_title("Employees Taking Leave Each Month")
# ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
# ax.grid(axis='y', linestyle='--', alpha=0.7)

# # Display plot in Streamlit
# st.pyplot(fig)




# # Calculate total leave days for each employee
# leave_df['Leave_Days'] = (leave_df['Leave_End'] - leave_df['Leave_Start']).dt.days + 1
# leave_summary = leave_df.groupby('Employee')['Leave_Days'].sum().reset_index()

# # Get the top 5 employees with the most leave taken
# top_5_most_leave = leave_summary.nlargest(5, 'Leave_Days')

# # Get the bottom 5 employees with the least leave taken
# bottom_5_least_leave = leave_summary.nsmallest(5, 'Leave_Days')

# # Display in Streamlit
# st.subheader("🏆 Employees with Most & Least Leave Taken")

# col1, col2 = st.columns(2)

# with col1:
#     st.write("**Top 5 Employees with Most Leave Taken**")
#     st.dataframe(top_5_most_leave)

# with col2:
#     st.write("**Bottom 5 Employees with Least Leave Taken**")
#     st.dataframe(bottom_5_least_leave)




# # Calculate Leave Days
# leave_df['Leave_Days'] = (leave_df['Leave_End'] - leave_df['Leave_Start']).dt.days + 1
# leave_summary = leave_df.groupby('Employee')['Leave_Days'].sum().reset_index()

# # Top 5 Employees with Most Leave Taken
# top_5_most_leave = leave_summary.nlargest(5, 'Leave_Days')

# # Bottom 5 Employees with Least Leave Taken
# bottom_5_least_leave = leave_summary.nsmallest(5, 'Leave_Days')

# # Streamlit Layout
# st.subheader("🏆 Employees with Most & Least Leave Taken")

# # Plot Top 5 Employees with Most Leave (Stacked)
# st.write("### 🏅 Top 5 Employees with Most Leave Taken")
# fig, ax = plt.subplots(figsize=(8, 5))
# sns.barplot(y=top_5_most_leave['Employee'], x=top_5_most_leave['Leave_Days'], palette='Blues_r', ax=ax)
# ax.set_xlabel("Total Leave Days")
# ax.set_ylabel("Employee")
# ax.set_title("Most Leave Taken")
# st.pyplot(fig)

# # Plot Bottom 5 Employees with Least Leave (Stacked)
# st.write("### 📉 Bottom 5 Employees with Least Leave Taken")
# fig, ax = plt.subplots(figsize=(8, 5))
# sns.barplot(y=bottom_5_least_leave['Employee'], x=bottom_5_least_leave['Leave_Days'], palette='Reds_r', ax=ax)
# ax.set_xlabel("Total Leave Days")
# ax.set_ylabel("Employee")
# ax.set_title("Least Leave Taken")
# st.pyplot(fig)




# # Calculate key leave metrics
# total_leaves = len(filtered_df)  # Total leave records
# total_leave_days = filtered_df['Leave_Days'].sum()  # Sum of all leave days
# average_leave_days = round(filtered_df['Leave_Days'].mean(), 2) if total_leaves > 0 else 0  # Avoid division by zero

# # Layout for displaying metrics
# col1, col2, col3 = st.columns(3)

# col1.metric(label="📊 Total Leave Applications", value=total_leaves)
# col2.metric(label="📅 Total Leave Days", value=total_leave_days)
# col3.metric(label="📊 Avg Leave Duration (Days)", value=average_leave_days)

# # Leave Type Breakdown
# with st.expander("📊 View Leave Type Breakdown"):
#  st.subheader("📊 Leave Type Breakdown")
#  col4, col5, col6, col7 = st.columns(4)
#  col4.metric(label="Annual Leave", value=total_leave["Annual"])
#  col5.metric(label="Sick Leave", value=total_leave["Sick"])
#  col6.metric(label="Casual Leave", value=total_leave["Casual"])
#  col7.metric(label="Maternity Leave", value=total_leave["Maternity"])

# # Group by Month_Name and Leave_Type
# # Group by Month_Name and Leave_Type
# leave_trend = filtered_df.groupby(['Month_Name', 'Leave_Type']).size().reset_index(name='Leave_Count')

# # Pivot for Stacked Bar Chart
# leave_pivot = leave_trend.pivot(index='Month_Name', columns='Leave_Type', values='Leave_Count').fillna(0)

# # Convert all values to numeric (avoiding non-numeric issues)
# # leave_pivot = leave_pivot.apply(pd.to_numeric, errors='coerce')

# # Reset index for easier viewing
# leave_pivot_reset = leave_pivot.reset_index()

# # Using expander to hide pivot table
# with st.expander("📋 View Leave Pivot Table"):
#     st.write(leave_pivot_reset)

# # Ensure the pivot table has data before plotting
# if leave_pivot.empty:
#     st.warning("⚠ No leave data available for the selected filters.")
# else:
#     # Display leave trend chart
#     st.subheader("📈 Leave Trend by Month & Type")
#     fig, ax = plt.subplots(figsize=(10, 5))
#     leave_pivot.plot(kind='bar', stacked=True, colormap='Set2', ax=ax)

#     ax.set_xlabel("Month")
#     ax.set_ylabel("Number of Leave Applications")
#     ax.set_title("Leave Trend by Month & Type")
#     ax.legend(title="Leave Type")
#     ax.grid(axis='y', linestyle='--', alpha=0.7)

#     st.pyplot(fig)







# import pandas as pd
# import streamlit as st
# import matplotlib.pyplot as plt
# import seaborn as sns

# # Sample leave application data
# data = {
#     'Employee': ['John', 'Sarah', 'Mike', 'Anna', 'David', 'Emily', 'Tom', 'Sophia', 'Chris', 'Emma'],
#     'Leave_Start': ['2024-01-10', '2024-02-15', '2024-02-20', '2024-03-05', '2024-03-18',
#                     '2024-04-25', '2024-04-30', '2024-05-10', '2024-02-25', '2024-04-12'],
#     'Leave_End': ['2024-01-12', '2024-02-18', '2024-02-22', '2024-03-07', '2024-03-22',
#                   '2024-04-28', '2024-05-02', '2024-05-12', '2024-02-28', '2024-04-15'],
#     'Leave_Type': ['Annual', 'Sick', 'Annual', 'Maternity', 'Casual', 'Sick', 'Annual', 'Casual', 'Sick', 'Annual']
# }

# # Convert to DataFrame
# leave_df = pd.DataFrame(data)

# # Convert date columns to datetime format
# leave_df['Leave_Start'] = pd.to_datetime(leave_df['Leave_Start'])
# leave_df['Leave_End'] = pd.to_datetime(leave_df['Leave_End'])

# # Calculate leave duration
# leave_df['Leave_Duration'] = (leave_df['Leave_End'] - leave_df['Leave_Start']).dt.days + 1

# # Extract Month Name from Leave_Start date
# leave_df['Month_Name'] = leave_df['Leave_Start'].dt.strftime('%B')

# # Define month order for sorting
# month_order = ['January', 'February', 'March', 'April', 'May', 'June', 
#                'July', 'August', 'September', 'October', 'November', 'December']

# # Convert Month_Name column to categorical
# leave_df['Month_Name'] = pd.Categorical(leave_df['Month_Name'], categories=month_order, ordered=True)

# # Streamlit app layout
# st.title("📊 Leave Analytics Dashboard")

# st.write("""
# This interactive dashboard provides insights into leave applications, 
# including trends, leave type distribution, and average leave durations.
# """)

# # Sidebar Filters
# st.sidebar.header("🔍 Filters")

# # Employee Filter
# selected_employee = st.sidebar.selectbox("Select Employee:", ["All"] + sorted(leave_df['Employee'].unique()))

# # Leave Type Filter
# selected_leave_types = st.sidebar.multiselect("Select Leave Type:", leave_df['Leave_Type'].unique(), default=leave_df['Leave_Type'].unique())

# # # Date Range Filter
# # min_date = leave_df['Leave_Start'].min()
# # max_date = leave_df['Leave_Start'].max()
# # start_date, end_date = st.sidebar.date_input("Select Date Range:", [min_date, max_date], min_value=min_date, max_value=max_date)

# # Apply Filters
# filtered_df = leave_df[
#     ((leave_df['Employee'] == selected_employee) | (selected_employee == "All")) &
#     (leave_df['Leave_Type'].isin(selected_leave_types))
# ]


# # Display filtered data table
# st.subheader("📋 Filtered Leave Data")
# st.dataframe(filtered_df)

# # ----- Analytics Section -----
# if not filtered_df.empty:
    
#     # --- Average Leave Duration ---
#     avg_duration = (filtered_df['Leave_Duration'].sum())
#     st.subheader("📅 Average Leave Duration")
#     st.metric(label="Average Leave Duration (Days)", value=avg_duration)

#     # --- Leave Type Distribution (Pie Chart) ---
#     st.subheader("🥧 Leave Type Distribution")
#     leave_type_counts = filtered_df['Leave_Type'].value_counts()

#     fig, ax = plt.subplots(figsize=(5, 5))
#     ax.pie(leave_type_counts, labels=leave_type_counts.index, autopct='%1.1f%%', colors=sns.color_palette("Set2"), startangle=140)
#     ax.set_title("Leave Type Distribution")
#     st.pyplot(fig)

#     # --- Leave Trend by Month ---
#     leave_trend = filtered_df.groupby(['Month_Name', 'Leave_Type']).size().reset_index(name='Leave_Count')

#     if not leave_trend.empty:
#         leave_pivot = leave_trend.pivot(index='Month_Name', columns='Leave_Type', values='Leave_Count').fillna(0)

#         st.subheader("📈 Leave Trend by Month & Type")
#         fig, ax = plt.subplots(figsize=(10, 5))
#         leave_pivot.plot(kind='bar', stacked=True, colormap='Set2', ax=ax)

#         ax.set_xlabel("Month")
#         ax.set_ylabel("Number of Leave Applications")
#         ax.set_title(f"Leave Trend by Month & Type ({selected_employee})")
#         ax.legend(title="Leave Type")
#         ax.grid(axis='y', linestyle='--', alpha=0.7)
#         st.pyplot(fig)

#     # --- Heatmap for Monthly Leave Trends ---
#     st.subheader("🔥 Monthly Leave Trend Heatmap")
#     heatmap_data = filtered_df.groupby(['Month_Name', 'Leave_Type'])['Leave_Duration'].sum().unstack().fillna(0)

#     fig, ax = plt.subplots(figsize=(10, 5))
#     sns.heatmap(heatmap_data, cmap="Blues", annot=True, fmt=".1f", linewidths=0.5, ax=ax)
#     ax.set_title("Total Leave Days by Month & Type")
#     st.pyplot(fig)

# else:
#     st.warning("⚠️ No data available for the selected filters.")







# BEST AND COMPLETE SELF-EXCLUSION ANALYSIS REPORT


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
# required_columns = ["OBSERVATION RATING", "NUMBER OF TIME", "USERS", "USER REASON", "PHONE", "REQUEST MEDIUM", "STATE", "DATE"]
# missing_columns = [col for col in required_columns if col not in df.columns]
# if missing_columns:
#     print(f"Error: Missing columns {missing_columns} in the data.")
#     exit()

# # Ensure phone numbers are treated as strings and preserve leading zeros
# df['PHONE'] = df['PHONE'].astype(str).str.zfill(11)

# # Total Requests
# total_requests = len(df)

# # Standardize the 'OBSERVATION RATING' column
# df["OBSERVATION RATING"] = df["OBSERVATION RATING"].str.lower()

# # Observation Rating Counts
# rating_counts = df["OBSERVATION RATING"].value_counts()

# # Filter Low and Medium observation ratings
# low_count_filtered = df[df["OBSERVATION RATING"] == 'low'][["USERS", "USER REASON", "NUMBER OF TIME", "PHONE"]]
# medium_count_filtered = df[df["OBSERVATION RATING"] == 'medium'][["USERS", "USER REASON", "NUMBER OF TIME", "PHONE"]]

# # Group by "REQUEST MEDIUM"
# grouped_medium = df.groupby("REQUEST MEDIUM").size().reset_index(name="Occurrence").sort_values(by="Occurrence", ascending=False)

# # Group by "USER REASON"
# grouped_by_reason = df.groupby("USER REASON").size().reset_index(name="Count").sort_values(by="Count", ascending=False)

# # Count Self-Excluded Employees by State & Include Employee_IDs
# state_counts = df.groupby("STATE").agg(
#     Count=("USERS", "count"),
#     USERS=("USERS", lambda x: list(x))
# ).reset_index()

# # Sort in descending order
# state_counts = state_counts.sort_values(by="Count", ascending=False)

# # Convert USERS list to string
# state_counts["USERS"] = state_counts["USERS"].apply(lambda x: ", ".join(map(str, x)))

# # Get the top 3 and bottom 3 states
# top_3_states = state_counts.nlargest(3, 'Count')
# bottom_3_states = state_counts.nsmallest(3, 'Count')
# combined_states = pd.concat([top_3_states, bottom_3_states])

# # Convert "DATE" column to datetime format
# df["DATE"] = pd.to_datetime(df["DATE"])
# daily_resolutions = df.groupby(df["DATE"].dt.date).size().reset_index(name="Resolutions Count").sort_values(by="Resolutions Count", ascending=False)

# # Convert rating_counts into a DataFrame with states as index and ratings as columns
# rating_counts_df = df.pivot_table(index="STATE", columns="OBSERVATION RATING", values="NUMBER OF TIME", aggfunc="sum", fill_value=0)

# # Correlation between Count and self-excluded users per state
# correlation = state_counts["Count"].corr(state_counts["Count"])

# # Find states with the highest and lowest 'Medium' ratings
# if "Medium" in rating_counts_df.columns:
#     most_medium_state = rating_counts_df["Medium"].idxmax()
#     least_medium_state = rating_counts_df["Medium"].idxmin()

# # Save results to TXT
# output_txt = "monthly-analysis-output.txt"
# with open(output_txt, "w") as f:
#     f.write(f"Total self-exclusion requests for the month: {total_requests}\n\n")
#     f.write("Observation Rating Counts:\n")
#     f.write(rating_counts.to_string() + "\n\n")
#     f.write("Requests by Medium:\n")
#     f.write(grouped_medium.to_string(index=False) + "\n\n")
#     f.write("Requests by User Reason:\n")
#     f.write(grouped_by_reason.to_string(index=False) + "\n\n")
#     f.write("State Counts:\n")
#     f.write(state_counts.to_string(index=False) + "\n\n")
#     f.write("Top and Bottom 3 States:\n")
#     f.write(f"Top 3 states: {top_3_states}\n")
#     f.write(f"least 3 states: {bottom_3_states}\n")


#     f.write(combined_states.to_string(index=False) + "\n\n")
#     f.write("Daily Resolutions:\n")
#     f.write(daily_resolutions.to_string(index=False) + "\n")
#     f.write(f"Correlation between Count and Self-Excluded Users: {correlation}\n")
    

# # Generate and save graphs
# graph_files = []

# # Observation Rating Bar Chart
# plt.figure(figsize=(8, 6))
# sns.barplot(x=rating_counts.index, y=rating_counts.values, palette="viridis")
# plt.title("Observation Rating Counts")
# plt.xlabel("Observation Rating")
# plt.ylabel("Count")
# bar_chart_file = "observation_rating_counts.png"
# plt.savefig(bar_chart_file)
# graph_files.append(bar_chart_file)
# plt.close()

# # Users Count by State Bar Chart
# plt.figure(figsize=(8, 6))
# sns.barplot(x="STATE", y="Count", data=state_counts, palette="viridis")
# plt.title("Users count by state")
# plt.xlabel("State")
# plt.ylabel("Number of Users")
# state_chart_file = "state_counts.png"
# plt.savefig(state_chart_file)
# graph_files.append(state_chart_file)
# plt.close()

# # Request Medium Pie Chart
# plt.figure(figsize=(8, 8))
# plt.pie(
#     grouped_medium["Occurrence"],
#     labels=grouped_medium["REQUEST MEDIUM"],
#     autopct="%1.1f%%",
#     startangle=140,
#     colors=sns.color_palette("pastel"),
# )
# plt.title("Distribution of Request Medium")
# pie_chart_file = "request_medium_distribution.png"
# plt.savefig(pie_chart_file)
# graph_files.append(pie_chart_file)
# plt.close()

# # User Reasons Bar Chart
# plt.figure(figsize=(10, 6))
# sns.barplot(x=grouped_by_reason["Count"], y=grouped_by_reason["USER REASON"], palette="coolwarm")
# plt.title("Top User Reasons")
# plt.xlabel("Count")
# plt.ylabel("User Reason")
# user_reason_chart_file = "top_user_reasons.png"
# plt.savefig(user_reason_chart_file)
# graph_files.append(user_reason_chart_file)
# plt.close()

# # Line Graph for Resolutions Over Time
# plt.figure(figsize=(10, 6))
# sns.lineplot(x=daily_resolutions["DATE"], y=daily_resolutions["Resolutions Count"], marker="o", color="blue")
# plt.title("Self-Exclusion Resolutions Per Day")
# plt.xlabel("Date")
# plt.ylabel("Resolution Count")
# plt.xticks(rotation=45)
# plt.grid(True)
# line_chart_file = "self_exclusion_resolutions.png"
# plt.savefig(line_chart_file)
# graph_files.append(line_chart_file)
# plt.close()

# # Heatmap Visualization
# plt.figure(figsize=(8, 5))
# sns.heatmap(rating_counts_df, annot=True, cmap="coolwarm", linewidths=0.5, fmt="d")
# plt.title("Self-Exclusion Ratings Heatmap")
# plt.xlabel("Observation Rating")
# plt.ylabel("State")
# heatmap_file = "ratings_heatmap.png"
# plt.savefig(heatmap_file)
# graph_files.append(heatmap_file)
# plt.close()









# GREAT SELF EXCLUSION REPORT IN DETAILS 2  GOOD ONE



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
# required_columns = ["OBSERVATION RATING", "NUMBER OF TIME", "USERS", "USER REASON", "PHONE", "REQUEST MEDIUM", "STATE", "DATE"]
# missing_columns = [col for col in required_columns if col not in df.columns]
# if missing_columns:
#     print(f"Error: Missing columns {missing_columns} in the data.")
#     exit()

# # Ensure phone numbers are treated as strings and preserve leading zeros
# df['PHONE'] = df['PHONE'].astype(str).str.zfill(11)

# # Total Requests
# total_requests = len(df)

# # Standardize the 'OBSERVATION RATING' column
# df["OBSERVATION RATING"] = df["OBSERVATION RATING"].str.lower()

# # Observation Rating Counts
# rating_counts = df["OBSERVATION RATING"].value_counts()

# # Filter Low and Medium observation ratings
# low_count_filtered = df[df["OBSERVATION RATING"] == 'low'][["USERS", "USER REASON", "NUMBER OF TIME", "PHONE"]]
# medium_count_filtered = df[df["OBSERVATION RATING"] == 'medium'][["USERS", "USER REASON", "NUMBER OF TIME", "PHONE"]]

# # Group by "REQUEST MEDIUM"
# grouped_medium = df.groupby("REQUEST MEDIUM").size().reset_index(name="Occurrence").sort_values(by="Occurrence", ascending=False)

# # Group by "USER REASON"
# grouped_by_reason = df.groupby("USER REASON").size().reset_index(name="Count").sort_values(by="Count", ascending=False)

# # Count Self-Excluded Employees by State & Include Employee_IDs
# state_counts = df.groupby("STATE").agg(
#     Count=("NUMBER OF TIME", "count"), 
#     USERS=("USERS", lambda x: list(x))
# ).reset_index()

# # Sort in descending order
# state_counts = state_counts.sort_values(by="Count", ascending=False)

# # Convert USERS list to string
# state_counts["USERS"] = state_counts["USERS"].apply(lambda x: ", ".join(map(str, x)))

# # Convert "DATE" column to datetime format
# df["DATE"] = pd.to_datetime(df["DATE"])
# daily_resolutions = df.groupby(df["DATE"].dt.date).size().reset_index(name="Resolutions Count").sort_values(by="Resolutions Count", ascending=False)

# # Save results to CSV and TXT

# output_txt = "monthly-analysis-output.txt"

# # df.to_csv(output_csv, index=False)

# with open(output_txt, "w") as f:
#     f.write(f"Total self-exclusion requests for the month: {total_requests}\n\n")
#     f.write("Observation Rating Counts:\n")
#     f.write(rating_counts.to_string() + "\n\n")
#     f.write("Requests by Medium:\n")
#     f.write(grouped_medium.to_string(index=False) + "\n\n")
#     f.write("Requests by User Reason:\n")
#     f.write(grouped_by_reason.to_string(index=False) + "\n\n")
#     f.write("State Counts:\n")
#     f.write(state_counts.to_string(index=False) + "\n\n")
#     f.write("Daily Resolutions:\n")
#     f.write(daily_resolutions.to_string(index=False) + "\n")

# # Generate and save graphs
# graph_files = []

# # Observation Rating Bar Chart
# plt.figure(figsize=(8, 6))
# sns.barplot(x=rating_counts.index, y=rating_counts.values, palette="viridis")
# plt.title("Observation Rating Counts")
# plt.xlabel("Observation Rating")
# plt.ylabel("Count")
# bar_chart_file = "observation_rating_counts.png"
# plt.savefig(bar_chart_file)
# graph_files.append(bar_chart_file)
# plt.close()

# # Users Count by State Bar Chart
# plt.figure(figsize=(8, 6))
# sns.barplot(x="STATE", y="Count", data=state_counts, palette="viridis")
# plt.title("Users count by state")
# plt.xlabel("State")
# plt.ylabel("Number of Users")
# bar_chart_file = "state_counts.png"
# plt.savefig(bar_chart_file)
# graph_files.append(bar_chart_file)
# plt.close()

# # Request Medium Pie Chart
# plt.figure(figsize=(8, 8))
# plt.pie(
#     grouped_medium["Occurrence"],
#     labels=grouped_medium["REQUEST MEDIUM"],
#     autopct="%1.1f%%",
#     startangle=140,
#     colors=sns.color_palette("pastel"),
# )
# plt.title("Distribution of Request Medium")
# pie_chart_file = "request_medium_distribution.png"
# plt.savefig(pie_chart_file)
# graph_files.append(pie_chart_file)
# plt.close()

# # User Reasons Bar Chart
# plt.figure(figsize=(10, 6))
# sns.barplot(x=grouped_by_reason["Count"], y=grouped_by_reason["USER REASON"], palette="coolwarm")
# plt.title("Top User Reasons")
# plt.xlabel("Count")
# plt.ylabel("User Reason")
# user_reason_chart_file = "top_user_reasons.png"
# plt.savefig(user_reason_chart_file)
# graph_files.append(user_reason_chart_file)
# plt.close()

# # Line Graph for Resolutions Over Time
# plt.figure(figsize=(10, 6))
# sns.lineplot(x=daily_resolutions["DATE"], y=daily_resolutions["Resolutions Count"], marker="o", color="blue")
# plt.title("Self-Exclusion Resolutions Per Day")
# plt.xlabel("Date")
# plt.ylabel("Resolution Count")
# plt.xticks(rotation=45)
# plt.grid(True)
# line_chart_file = "self_exclusion_resolutions.png"
# plt.savefig(line_chart_file)
# graph_files.append(line_chart_file)
# plt.close()

# print(f"Analysis saved to '{output_txt}")
# print(f"Generated graphs: {', '.join(graph_files)}")








# SELF EXCLUSION REPORT IN DETAILS 1

# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
# import os
# import openpyxl

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
# df['PHONE'] = df['PHONE'].astype(str).str.zfill(11)  # Adjust to the required length

# # Total Requests
# total_requests = len(df)
# print(f"The Total self-exclusion requests for the month: {total_requests}\n")

# # Standardize the 'OBSERVATION RATING' column to lowercase
# df["OBSERVATION RATING"] = df["OBSERVATION RATING"].str.lower()

# # Observation Rating Counts
# rating_counts = df["OBSERVATION RATING"].value_counts()

# # Filter Low and Medium observation ratings
# low_count_filtered = df[df["OBSERVATION RATING"] == 'low'][["USERS", "USER REASON", "NUMBER OF TIME", "PHONE"]]
# medium_count_filtered = df[df["OBSERVATION RATING"] == 'medium'][["USERS", "USER REASON", "NUMBER OF TIME", "PHONE"]]

# # Group by "REQUEST MEDIUM"
# grouped_medium = df.groupby("REQUEST MEDIUM").size().reset_index(name="Occurrence").sort_values(by="Occurrence", ascending=False)

# # Group by "USER REASON"
# grouped_by_reason = df.groupby("USER REASON").size().reset_index(name="Count").sort_values(by="Count", ascending=False)


# # **Count Self-Excluded Employees by State & Include Employee_IDs**
# state_counts = df.groupby("STATE").agg(
#     Count=("NUMBER OF TIME", "count"), 
#     USERS=("USERS", lambda x: list(x))
# ).reset_index()

# # Sort in descending order
# state_counts = state_counts.sort_values(by="Count", ascending=False)

# # Convert Employee_ID list to string for better display
# state_counts["USERS"] = state_counts["USERS"].apply(lambda x: ", ".join(map(str, x)))



# # Ensure the "DATE" column is in datetime format
# if "DATE" in df.columns:
#     df["DATE"] = pd.to_datetime(df["DATE"])  # Convert to datetime
#     daily_resolutions = df.groupby(df["DATE"].dt.date).size().reset_index(name="Resolutions Count").sort_values(by="Resolutions Count", ascending=False)

# else:
#     print("Warning: 'DATE' column not found. Skipping daily resolution analysis.")
#     daily_resolutions = None

# # Save results to Excel and include graphs
# output_file = "monthly-analysis-output.xlsx"
# with pd.ExcelWriter(output_file, engine='xlsxwriter') as writer:
#     df.to_excel(writer, sheet_name="Main Data", index=False)
#     pd.DataFrame({"Total Requests": [total_requests]}).to_excel(writer, sheet_name="Summary", index=False)
#     rating_counts.to_frame().to_excel(writer, sheet_name="Rating Counts")
#     grouped_medium.to_excel(writer, sheet_name="Request Medium", index=False)
#     grouped_by_reason.to_excel(writer, sheet_name="User Reasons", index=False)
#     low_count_filtered.to_excel(writer, sheet_name="Low Ratings", index=False)
#     medium_count_filtered.to_excel(writer, sheet_name="Medium Ratings", index=False)
#     state_counts.to_excel(writer, sheet_name="state_countss", index=False)

#     # Save daily resolutions if available
#     if daily_resolutions is not None:
#         daily_resolutions.to_excel(writer, sheet_name="Daily Resolutions", index=False)

#     # Access the workbook and create a worksheet for graphs
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




#     plt.figure(figsize=(8, 6))
#     sns.barplot(x="STATE", y="Count", data=state_counts, palette="viridis")
#     plt.title("Users count by state")
#     plt.xlabel("state")
#     plt.ylabel("Number of users")
#     bar_chart_file = "state_counts.png"
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
#     sns.barplot(x=grouped_by_reason["Count"], y=grouped_by_reason["USER REASON"], palette="coolwarm")
#     plt.title("Top User Reasons")
#     plt.xlabel("Count")
#     plt.ylabel("User Reason")
#     user_reason_chart_file = "top_user_reasons.png"
#     plt.savefig(user_reason_chart_file)
#     graph_files.append(user_reason_chart_file)
#     plt.close()

    # # Line Graph for Resolutions Over Time
    # if daily_resolutions is not None:
    #     plt.figure(figsize=(10, 6))
    #     sns.lineplot(x=daily_resolutions["DATE"], y=daily_resolutions["Resolutions Count"], marker="o", color="blue")
    #     plt.title("Self-Exclusion Resolutions Per Day")
    #     plt.xlabel("Date")
    #     plt.ylabel("Resolution Count")
    #     plt.xticks(rotation=45)
    #     plt.grid(True)

    #     line_chart_file = "self_exclusion_resolutions.png"
    #     plt.savefig(line_chart_file)
    #     graph_files.append(line_chart_file)
    #     plt.close()

#     # Insert images into the "Graphs" worksheet
#     for i, graph_file in enumerate(graph_files):
#         worksheet.insert_image(f"A{i * 20 + 1}", graph_file)

# # Clean up temporary graph files
# for graph_file in graph_files:
#     os.remove(graph_file)

# print(f"Data and graphs have been saved to '{output_file}'")















# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns

# # Set theme for seaborn
# sns.set_theme(style="whitegrid")

# # Sample weekly call report data
# data = {
#     "Department": ["Customer Service Dept", "Commercial Dept"],
#     "Total_Calls_Offered": [8930, 61],
#     "Disconnected_Calls": [3047, 26],
#     "Total_Incoming_Calls": [5883, 35],
#     "Answered_First_Call": [5687, 29],
#     "Abandoned_First_Call": [193, 6],
#     "We_Called_Back": [0, 0],
#     "Actual_Abandoned": [0, 0],  # No abandoned calls recorded
#     "Total_Answered_Calls": [5687, 29],
#     "Avg_Waiting_Time_Minutes": [0.2, 0.15],  # Converted from seconds to minutes (12s = 0.2 min, 9s = 0.15 min)
#     "Success_Ratio": [96.67, 82.86]
# }

# # Create DataFrame
# df = pd.DataFrame(data)

# # Additional KPI Calculations
# df["Abandonment_Rate"] = (df["Actual_Abandoned"] / df["Total_Calls_Offered"]) * 100
# df["FCR_Rate"] = (df["Answered_First_Call"] / df["Total_Calls_Offered"]) * 100
# df["Call_Drop_Rate"] = (df["Disconnected_Calls"] / df["Total_Calls_Offered"]) * 100

# # Peak Hour and Busy Days Data
# peak_data = {
#     "Time_Range": ["8:00 AM - 3:00 PM"],
#     "Total_Offered_Calls": [5166],
#     "Busiest_Day_1": ["Monday"],
#     "Calls_On_Busiest_Day_1": [1487],
#     "Busiest_Day_2": ["Wednesday"],
#     "Calls_On_Busiest_Day_2": [1408],
# }

# peak_df = pd.DataFrame(peak_data)

# # --- Visualization 1: Call Volume by Department ---
# plt.figure(figsize=(8, 5))
# sns.barplot(x=df["Department"], y=df["Total_Calls_Offered"], palette="Blues_r")
# plt.title("Total Calls Offered by Department")
# plt.ylabel("Number of Calls")
# plt.xticks(rotation=45)
# plt.show()

# # --- Visualization 2: Success Ratio vs Abandonment Rate ---
# plt.figure(figsize=(8, 5))
# sns.scatterplot(x=df["Success_Ratio"], y=df["Abandonment_Rate"], hue=df["Department"], s=100)
# plt.title("Success Ratio vs. Abandonment Rate")
# plt.xlabel("Success Ratio (%)")
# plt.ylabel("Abandonment Rate (%)")
# plt.show()

# # --- Automated Summary Report ---
# report = f"""
# 📊 **Weekly Call Center Performance Report**

# ### **1️⃣ Key Performance Metrics**
# 🔹 **Total Calls Offered:** {df["Total_Calls_Offered"].sum()}  
# 🔹 **Total Answered Calls:** {df["Total_Answered_Calls"].sum()}  
# 🔹 **Overall Success Ratio:** {round(df["Success_Ratio"].mean(), 2)}%  
# 🔹 **Average Waiting Time:** {round(df["Avg_Waiting_Time_Minutes"].mean(), 2)} minutes  

# ### **2️⃣ Departmental Breakdown**
# **Customer Service Dept:**  
# ✅ Total Calls Offered: {df.iloc[0]["Total_Calls_Offered"]}  
# ✅ Success Ratio: {df.iloc[0]["Success_Ratio"]}%  
# ✅ First Call Resolution Rate: {round(df.iloc[0]["FCR_Rate"], 2)}%  
# ✅ Average Waiting Time: {df.iloc[0]["Avg_Waiting_Time_Minutes"]} minutes  

# **Commercial Dept:**  
# ✅ Total Calls Offered: {df.iloc[1]["Total_Calls_Offered"]}  
# ✅ Success Ratio: {df.iloc[1]["Success_Ratio"]}%  
# ✅ First Call Resolution Rate: {round(df.iloc[1]["FCR_Rate"], 2)}%  
# ✅ Average Waiting Time: {df.iloc[1]["Avg_Waiting_Time_Minutes"]} minutes  

# ### **3️⃣ Peak Hour & Busy Days**
# 📌 **Busiest Time:** {peak_df["Time_Range"][0]} ({peak_df["Total_Offered_Calls"][0]} calls)  
# 📌 **Most Busy Days:**  
# 🔸 {peak_df["Busiest_Day_1"][0]} - {peak_df["Calls_On_Busiest_Day_1"][0]} calls  
# 🔸 {peak_df["Busiest_Day_2"][0]} - {peak_df["Calls_On_Busiest_Day_2"][0]} calls  

# ### **4️⃣ Key Observations**
# 🚀 **Performance:** The overall success ratio is high, with an efficient call handling process.  
# ⚠️ **Network Issues:** Some CCRs experienced login difficulties due to MTN & Airtel network problems.  
# 📞 **Call Drops:** Disconnection rates remain a concern, especially in the Commercial Department.  

# ### **5️⃣ Recommendations**
# 📍 Strengthen **network infrastructure** to prevent login issues.  
# 📍 Reduce **call drop rates** by improving VoIP stability.  
# 📍 Increase **staffing during peak hours** to manage higher call volumes.  

# ---
# **End of Report**
# """

# # Print the generated report
# print(report)






# METRIC


# import streamlit as st
# import pandas as pd

# # --- Sample Customer Support Data ---
# data = {
#     "Ticket ID": ["T1001", "T1002", "T1003", "T1004", "T1005", "T1006", "T1007", "T1008"],
#     "Customer Issue": [
#         "Delayed Withdrawal", "Account Frozen", "Bonus Not Credited", 
#         "Wrong Bet Settlement", "Self-Exclusion Request", "Card Removal Issue",
#         "Verification Pending", "Deposit Not Reflecting"
#     ],
#     "Status": ["Resolved", "Pending", "Resolved", "Resolved", "Pending", "Resolved", "Pending", "Resolved"],
#     "Resolution Time (Hours)": [48, 72, 12, 24, 168, 36, 48, 10],
#     "Customer Rating": [4, None, 5, 3, None, 5, None, 4]
# }
# df = pd.DataFrame(data)

# # --- Key Metrics ---
# resolved_tickets = df[df["Status"] == "Resolved"].shape[0]
# pending_tickets = df[df["Status"] == "Pending"].shape[0]
# avg_resolution_time = df["Resolution Time (Hours)"].mean()

# # --- Layout with Custom Columns ---
# col1, col2, col3 = st.columns(3)

# # Customizing metrics with delta, color, and text labels
# col1.metric(
#     label="✅ Resolved Tickets", 
#     value=resolved_tickets,
#     delta=resolved_tickets - 5,  # Example delta for change, can be dynamic
#     delta_color="inverse",  # Use 'inverse' for red or 'normal' for green
#     help="Total number of tickets resolved within the last month"
# )

# col2.metric(
#     label="⏳ Pending Tickets", 
#     value=pending_tickets,
#     delta=pending_tickets - 3,
#     delta_color="normal",  # Green color for increase
#     help="Tickets that are yet to be resolved"
# )

# col3.metric(
#     label="📉 Avg Resolution Time (Hours)", 
#     value=round(avg_resolution_time, 2),
#     delta=round(avg_resolution_time - 5, 2),  # Example delta for change
#     delta_color="normal",  # Green color for improvement
#     help="Average time taken to resolve complaints in hours"
# )

# # --- Show Data ---
# st.subheader("🔍 Customer Complaints Data")
# st.write(df)








# # INTERACTIVE LOGIN PAGE

# import streamlit as st
 
# def login():
#     st.title("Login")
#     username = st.text_input("Username")
#     password = st.text_input("Pasword", type="password")
#     if st.button("login"):
#         if username == "admin" and password == "password":
#             st.session_state.logged_in=True
#             st.success("Login successful")
#         else:
#             st.error("Invalid username and password is incorrect")   

# def dashboard():
#     st.title("Dashboard")
#     st.write("Welcome admin")
#     if st.button("sign-out"):
#         st.session_state.logged_in=False
#         st.rerun()


# if "logged_in" not in st.session_state:
#     st.session_state.logged_in=False


# if st.session_state.logged_in:
    
#     dashboard()

# else:
#     login()    












# # AUTHENTICATION LOGIN PAGE


# import streamlit as st
# import pandas as pd

# # --- Sample Customer Support Data ---
# data = {
#     "Ticket ID": ["T1001", "T1002", "T1003", "T1004", "T1005", "T1006", "T1007", "T1008"],
#     "Customer Issue": [
#         "Delayed Withdrawal", "Account Frozen", "Bonus Not Credited", 
#         "Wrong Bet Settlement", "Self-Exclusion Request", "Card Removal Issue",
#         "Verification Pending", "Deposit Not Reflecting"
#     ],
#     "Status": ["Resolved", "Pending", "Resolved", "Resolved", "Pending", "Resolved", "Pending", "Resolved"],
#     "Resolution Time (Hours)": [48, 72, 12, 24, 168, 36, 48, 10],
#     "Customer Rating": [4, None, 5, 3, None, 5, None, 4]
# }
# df = pd.DataFrame(data)

# # --- User Authentication ---
# USER_CREDENTIALS = {
#     "manager": "bet9ja2024",
#     "analyst": "data123"
# }

# def authenticate(username, password):
#     return USER_CREDENTIALS.get(username) == password

# if "authenticated" not in st.session_state:
#     st.session_state.authenticated = False

# # --- Login Form ---
# if not st.session_state.authenticated:
#     st.sidebar.header("Login")
#     username = st.sidebar.text_input("Username", key="user")
#     password = st.sidebar.text_input("Password", type="password", key="pass", help="Enter your secure password")
#     login_btn = st.sidebar.button("Login")

#     if login_btn:
#         if authenticate(username, password):
#             st.session_state.authenticated = True
#             st.session_state.username = username
#             st.sidebar.success(f"✅ Welcome, {username}!")
#             st.rerun()  # Corrected line
#         else:
#             st.sidebar.error("❌ Invalid username or password")

# # --- Main Dashboard: Show Data Only If Logged In ---
# if st.session_state.authenticated:
#     st.title("📊 Bet9ja Customer Support Analytics")

#     # Display Data
#     st.subheader("🔍 Customer Complaints Data")
#     st.write(df)

#     # Metrics
#     resolved_tickets = df[df["Status"] == "Resolved"].shape[0]
#     pending_tickets = df[df["Status"] == "Pending"].shape[0]
#     avg_resolution_time = df["Resolution Time (Hours)"].mean()

#     col1, col2, col3 = st.columns(3)
#     col1.metric("✅ Resolved Tickets", resolved_tickets)
#     col2.metric("⏳ Pending Tickets", pending_tickets)
#     col3.metric("📉 Avg Resolution Time (Hours)", round(avg_resolution_time, 2))

#     # Data Visualization
#     st.subheader("📈 Resolution Time Analysis")
#     st.bar_chart(df.set_index("Customer Issue")["Resolution Time (Hours)"])

#     # Customer Rating Distribution
#     st.subheader("🌟 Customer Ratings Breakdown")
#     rating_counts = df["Customer Rating"].value_counts().sort_index()
#     st.bar_chart(rating_counts)

#     # Logout Button
#     if st.sidebar.button("Logout"):
#         st.session_state.authenticated = False
#         st.rerun()  # Corrected line





#  GREAT CRM DASHBOARD WITH EMOGI AND LOGIN PAGEGOOD FOR THE DEMO  2




# import streamlit as st
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns

# # Define login function
# def login():
#     st.title("Login")
#     username = st.text_input("Username")
#     password = st.text_input("Password", type="password")
#     if st.button("Login"):
#         if username == "user" and password == "myuser":
#             st.session_state.logged_in = True
#             st.success("Login successful")
#             st.rerun()
#         else:
#             st.error("Invalid username and password is incorrect")

# # Define dashboard function
# def dashboard():
#     st.title("Case Management Dashboard For KCG 🧑‍💻")
#     st.image('https://streamlit.io/images/brand/streamlit-mark-color.svg', width=150, caption='Streamlit Dashboard 🎯')
#     st.markdown("""
#         Welcome to the **Case Management Dashboard**! 📊
#         This dashboard helps track cases, statuses, and issues efficiently. Use the filters to explore detailed case data based on the selected categories. 🕵️‍♂️
#     """)
    
#     # Load dataset
#     filepath = "CASES.xlsx"
#     try:
#         df = pd.read_excel(filepath, engine='openpyxl')
#     except Exception as e:
#         st.error(f"Error loading Excel file: {e}")
#         st.stop()
    
#     required_columns = ['Date Created', 'Date Modified', 'Assigned to', 'Case Number', 'Status', 'Online Users', 'General complaint', 'Agent']
#     missing_columns = [col for col in required_columns if col not in df.columns]
#     if missing_columns:
#         st.error(f"Missing columns: {', '.join(missing_columns)}")
#         st.stop()
    
#     df['Date Created'] = pd.to_datetime(df['Date Created'], errors='coerce')
#     df['Date Modified'] = pd.to_datetime(df['Date Modified'], errors='coerce')
#     df = df.dropna(subset=['Date Created', 'Date Modified'])
#     df['Time Taken'] = df['Date Modified'] - df['Date Created']
#     df['Time Taken (Readable)'] = df['Time Taken'].apply(lambda x: f"{x.days} days, {x.seconds // 3600} hours, {(x.seconds // 60) % 60} minutes")
#     df['Merged'] = df['Online Users'].fillna('') + ' ' + df['General complaint'].fillna('') + ' ' + df['Agent'].fillna('')
#     df_cleaned = df.drop(columns=['Online Users', 'General complaint', 'Agent'])
    
#     # Sidebar Filters
#     st.sidebar.title("Filters 🔍")
#     assigned_to_search = st.sidebar.selectbox('Search by Assigned to:', df['Assigned to'].unique())
#     status_filter = st.sidebar.selectbox('Filter by Status:', ['All', 'Opened', 'In Progress', 'Resolved', 'Escalated'])
#     df = df[df['Assigned to'] == assigned_to_search]
#     if status_filter != 'All':
#         df = df[df['Status'] == status_filter]
#     st.sidebar.subheader(f"Filtered Cases for {assigned_to_search}")
#     st.sidebar.write(df[['Case Number', 'Status', 'Time Taken (Readable)']])
    
#     # Case Status Summary
#     st.subheader("Case Status Summary 📝")
#     case_status_summary = df['Status'].value_counts()
#     total_cases = {
#         "Escalated": case_status_summary.get("Escalated", 0),
#         "Resolved": case_status_summary.get("Resolved", 0),
#         "In Progress": case_status_summary.get("In Progress", 0),
#         "Opened": case_status_summary.get("Opened", 0)
#     }
#     st.write(f"**Total Escalated Cases**: {total_cases['Escalated']} 🔥")
#     st.write(f"**Total Resolved Cases**: {total_cases['Resolved']} ✅")
#     st.write(f"**Total In Progress Cases**: {total_cases['In Progress']} 🛠️")
#     st.write(f"**Total Opened Cases**: {total_cases['Opened']} 📂")
    
#     # Case Status Distribution
#     st.subheader("Case Status Distribution 📊")
#     fig, ax = plt.subplots(figsize=(10, 6))
#     sns.barplot(x=case_status_summary.index, y=case_status_summary.values, ax=ax, palette="viridis")
#     ax.set_title('Distribution of Case Statuses')
#     ax.set_ylabel('Number of Cases')
#     st.pyplot(fig)
    
#     # Sign-out Button
#     if st.button("Sign-out"):
#         st.session_state.logged_in = False
#         st.rerun()

# # Initialize session state for authentication
# if "logged_in" not in st.session_state:
#     st.session_state.logged_in = False

# # Load dashboard if logged in, otherwise show login page
# if st.session_state.logged_in:
#     dashboard()
# else:
#     login()














#  GREAT CRM DASHBOARD WITH EMOGI AND LOGIN PAGEGOOD FOR THE DEMO  1


# import streamlit as st
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns

# # Define user credentials
# USER_CREDENTIALS = {"admin": "password123", "user": "userpass"}

# # Initialize session state for authentication if not already set
# if "authenticated" not in st.session_state:
#     st.session_state.authenticated = False

# # Login Page
# if not st.session_state.authenticated:
#     st.title("Login Page 🔑")
#     username = st.text_input("Username")
#     password = st.text_input("Password", type="password")
    
#     if st.button("Login"):
#         if username in USER_CREDENTIALS and USER_CREDENTIALS[username] == password:
#             st.session_state.authenticated = True
#             st.rerun() 
#         else:
#             st.error("Invalid username or password. Please try again.")
    
#     st.stop()  # Stop further execution if not authenticated

# # Main Dashboard (only accessible after login)
# st.set_page_config(page_title="Case Management Dashboard", layout="wide", page_icon="📊")
# st.image('https://streamlit.io/images/brand/streamlit-mark-color.svg', width=150, caption='Streamlit Dashboard 🎯')
# st.title("Case Management Dashboard For KCG 🧑‍💻")

# # Load Excel File
# filepath = "CASES.xlsx"
# try:
#     df = pd.read_excel(filepath, engine='openpyxl')
# except Exception as e:
#     st.error(f"Error loading Excel file: {e}")
#     st.stop()

# # Ensure necessary columns exist
# required_columns = ['Date Created', 'Date Modified', 'Assigned to', 'Case Number', 'Status', 'Online Users', 'General complaint', 'Agent']
# missing_columns = [col for col in required_columns if col not in df.columns]
# if missing_columns:
#     st.error(f"Missing columns: {', '.join(missing_columns)}")
#     st.stop()

# # Convert dates and calculate time differences
# df['Date Created'] = pd.to_datetime(df['Date Created'], errors='coerce')
# df['Date Modified'] = pd.to_datetime(df['Date Modified'], errors='coerce')
# df = df.dropna(subset=['Date Created', 'Date Modified'])
# df['Time Taken'] = df['Date Modified'] - df['Date Created']
# df['Time Taken (Readable)'] = df['Time Taken'].apply(lambda x: f"{x.days} days, {x.seconds // 3600} hours, {(x.seconds // 60) % 60} minutes")

# # Merge issue-related columns
# df['Merged'] = df['Online Users'].fillna('') + ' ' + df['General complaint'].fillna('') + ' ' + df['Agent'].fillna('')
# df_cleaned = df.drop(columns=['Online Users', 'General complaint', 'Agent'])

# # Sidebar Filters
# st.sidebar.title("Filters 🔍")
# assigned_to_search = st.sidebar.selectbox('Search by Assigned to:', df['Assigned to'].unique())
# status_filter = st.sidebar.selectbox('Filter by Status:', ['All', 'Opened', 'In Progress', 'Resolved', 'Escalated'])

# df = df[df['Assigned to'] == assigned_to_search]
# if status_filter != 'All':
#     df = df[df['Status'] == status_filter]

# st.sidebar.subheader(f"Filtered Cases for {assigned_to_search}")
# st.sidebar.write(df[['Case Number', 'Status', 'Time Taken (Readable)']])

# # Case Status Summary
# st.subheader("Case Status Summary 📝")
# case_status_summary = df['Status'].value_counts()
# total_cases = {status: case_status_summary.get(status, 0) for status in ["Escalated", "Resolved", "In Progress", "Opened"]}

# for key, value in total_cases.items():
#     st.write(f"**Total {key} Cases**: {value}")

# # Case Status Distribution
# st.subheader("Case Status Distribution 📊")
# fig, ax = plt.subplots(figsize=(10, 6))
# sns.barplot(x=case_status_summary.index, y=case_status_summary.values, ax=ax, palette="viridis")
# ax.set_title('Distribution of Case Statuses')
# ax.set_ylabel('Number of Cases')
# st.pyplot(fig)

# # Issue Count Over Time
# st.subheader("Issue Count Over Time ⏳")
# df_cleaned_non_na = df_cleaned[df_cleaned['Merged'].notna() & (df_cleaned['Merged'] != '')]
# issue_count_by_date = df_cleaned_non_na.groupby(df['Date Created'].dt.date)['Merged'].count()

# fig2, ax2 = plt.subplots(figsize=(10, 6))
# issue_count_by_date.plot(kind='line', ax=ax2, color='blue', marker='o')
# ax2.set_title('Issue Count Over Time')
# ax2.set_xlabel('Date')
# ax2.set_ylabel('Issue Count')
# st.pyplot(fig2)

# # Logout Button
# if st.sidebar.button("Logout"):
#     st.session_state.authenticated = False
#     st.rerun() 








# GREAT CRM DASHBOARD WITH EMOGI GOOD FOR THE DEMO


# import streamlit as st
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns


# # Filepath
# filepath = "CASES.xlsx"

# # Create the DataFrame with error handling
# try:
#     df = pd.read_excel(filepath, engine='openpyxl')  # Use 'openpyxl' for reading Excel files
# except Exception as e:
#     st.error(f"Error loading Excel file: {e}")
#     st.stop()  # Stop further execution if the file can't be loaded

# # Ensure necessary columns exist
# required_columns = ['Date Created', 'Date Modified', 'Assigned to', 'Case Number', 'Status', 'Online Users', 'General complaint', 'Agent']
# missing_columns = [col for col in required_columns if col not in df.columns]
# if missing_columns:
#     st.error(f"Missing columns: {', '.join(missing_columns)}")
#     st.stop()

# # Convert 'Date Created' and 'Date Modified' to datetime with error handling
# df['Date Created'] = pd.to_datetime(df['Date Created'], errors='coerce')  # Coerce errors if any
# df['Date Modified'] = pd.to_datetime(df['Date Modified'], errors='coerce')

# # Drop rows with NaT (invalid datetime)
# df = df.dropna(subset=['Date Created', 'Date Modified'])

# # Calculate time difference between 'Date Created' and 'Date Modified'
# df['Time Taken'] = df['Date Modified'] - df['Date Created']

# # Human-readable time difference
# df['Time Taken (Readable)'] = df['Time Taken'].apply(lambda x: f"{x.days} days, {x.seconds // 3600} hours, {(x.seconds // 60) % 60} minutes")

# # Merge issue-related columns
# df['Merged'] = df['Online Users'].fillna('') + ' ' + df['General complaint'].fillna('') + ' ' + df['Agent'].fillna('')

# # Drop unnecessary columns
# df2 = df.drop(columns=['Online Users', 'General complaint', 'Agent'])




# # Streamlit UI Setup
# st.set_page_config(page_title="Case Management Dashboard", layout="wide", page_icon="📊")



# # Add the Streamlit logo and some emojis to enhance the UI
# st.image('https://streamlit.io/images/brand/streamlit-mark-color.svg', width=150, caption='Streamlit Dashboard 🎯')
# st.title("Case Management Dashboard For KCG 🧑‍💻")
# st.markdown("""
#     Welcome to the **Case Management Dashboard**! 📊
#     This dashboard helps track cases, statuses, and issues efficiently. Use the filters to explore detailed case data based on the selected categories. 🕵️‍♂️
# """)

# # Sidebar for Filtering
# st.sidebar.title("Filters 🔍")
# assigned_to_search = st.sidebar.selectbox('Search by Assigned to:', df['Assigned to'].unique())
# status_filter = st.sidebar.selectbox('Filter by Status:', ['All', 'Opened', 'In Progress', 'Resolved', 'Escalated'])


# # Apply status filter if not "All"
# if status_filter != 'All':
#     df = df[df['Status'] == status_filter]

# # Display filtered results based on 'Assigned to' and 'Status'
# st.sidebar.subheader(f"Filtered Cases for {assigned_to_search}")
# st.sidebar.write(df[['Case Number', 'Merged', 'Status', 'Time Taken (Readable)']])


# # Main Content: Case Status Summary
# st.subheader("Case Status Summary 📝")
# case_status_summary = df['Status'].value_counts()

# # st.write(case_status_summary)

# # Display total counts for each case status for the selected "Assigned to"
# total_cases = {
#     "Escalated": case_status_summary.get("Escalated", 0),
#     "Resolved": case_status_summary.get("Resolved", 0),
#     "In Progress": case_status_summary.get("In Progress", 0),
#     "Opened": case_status_summary.get("Opened", 0)
# }

# st.write(f"**Total Escalated Cases**: {total_cases['Escalated']} 🔥")
# st.write(f"**Total Resolved Cases**: {total_cases['Resolved']} ✅")
# st.write(f"**Total In Progress Cases**: {total_cases['In Progress']} 🛠️")
# st.write(f"**Total Opened Cases**: {total_cases['Opened']} 📂")

# # Main Content: Case Status Distribution
# st.subheader("Case Status Distribution 📊")
# fig, ax = plt.subplots(figsize=(10, 6))
# sns.barplot(x=case_status_summary.index, y=case_status_summary.values, ax=ax, palette="viridis")
# ax.set_title('Distribution of Case Statuses')
# ax.set_ylabel('Number of Cases')
# st.pyplot(fig)



# # Main Content: Count of Each Status by Assigned to
# st.subheader("Count of Each Status by Assigned to 📑")
# status_by_assigned = df.groupby(['Assigned to', 'Status']).size().reset_index(name='Count')
# st.write(status_by_assigned)



# # Main Content: Issue Count Over Time

# st.subheader("Issue Count Over Time ⏳")

# df['Date Created'] = df['Date Created'].dt.date

# df = df[df['Merged'].notna() & (df['Merged'] != '')]
# issue_count_by_date = df.groupby(['Date Created'])['Merged'].count()

# st.write(issue_count_by_date)

# fig2, ax2 = plt.subplots(figsize=(10, 6))
# issue_count_by_date.plot(kind='line', ax=ax2, color='blue', marker='o')
# ax2.set_title('Issue Count Over Time')
# ax2.set_xlabel('Date')
# ax2.set_ylabel('Issue Count')
# st.pyplot(fig2)




# # Display the total count of each issue
# st.subheader("Total Count of Each Issue  🔢")
# issue_counts = df['Merged'].value_counts()

# st.write(issue_counts)



# # Issue Count Section
# st.subheader("Total Count of Each Issue  🔢")

# # Count the unique issues in the 'Merged' column
# if not df['Merged'].isna().all():
#     issue_counts = df['Merged'].value_counts()  # Count occurrences of each issue

#     # Display the issue counts as a DataFrame for clarity
#     issue_counts_df = issue_counts.reset_index()
#     issue_counts_df.columns = ['Issue Description', 'Number of Cases']
#     st.write(issue_counts_df)

#     # Add a bar chart to visually represent issue counts
#     st.subheader("Issue Count Distribution 📊")
#     fig, ax = plt.subplots(figsize=(12, 6))
#     sns.barplot(data=issue_counts_df.head(10), x='Number of Cases', y='Issue Description', palette='magma', ax=ax)
#     ax.set_title('Top 10 Most Frequent Issues')
#     ax.set_xlabel('Number of Cases')
#     ax.set_ylabel('Issue Description')
#     st.pyplot(fig)
# else:
#     st.write("No issues found in the dataset.")




# # Issue Count Section
# st.subheader("Total Count of ALL Issue 🔢")

# # Count the unique issues in the 'Merged' column
# if not df2['Merged'].isna().all():
#     issue_counts = df2['Merged'].value_counts()  # Count occurrences of each issue

#     # Display the issue counts as a DataFrame for clarity
#     issue_counts_df = issue_counts.reset_index()
#     issue_counts_df.columns = ['Issue Description', 'Number of Cases']
#     st.write(issue_counts_df)

#     # Add a bar chart to visually represent issue counts
#     st.subheader("Issue Count Distribution 📊")
#     fig, ax = plt.subplots(figsize=(12, 6))
#     sns.barplot(data=issue_counts_df.head(10), x='Number of Cases', y='Issue Description', palette='magma', ax=ax)
#     ax.set_title('Top 10 Most Frequent Issues')
#     ax.set_xlabel('Number of Cases')
#     ax.set_ylabel('Issue Description')
#     st.pyplot(fig)
# else:
#     st.write("No issues found in the dataset.")




# # Section to get Top 5 Employees with Most Resolved Cases and Least Resolved Cases
# resolved_cases_df = df2[df2['Status'] == 'Resolved']
# resolved_cases_count = resolved_cases_df.groupby('Assigned to').size().reset_index(name='Resolved Cases')


# # Get the Top 5 Employees with the Most Resolved Cases
# top_5_resolved = resolved_cases_count.nlargest(5, 'Resolved Cases')

# # Get the Bottom 5 Employees with the Least Resolved Cases
# bottom_5_resolved = resolved_cases_count.nsmallest(5, 'Resolved Cases')

# # Display the results
# st.subheader("Top 5 Employees with the Most Resolved Cases 🏆")
# st.write(top_5_resolved)

# st.subheader("Bottom 5 Employees with the Least Resolved Cases 👎")
# st.write(bottom_5_resolved)

# # Section to get Top 5 Employees with Most Opened Cases and Bottom 5 Employees with Least Opened Cases
# opened_cases_df = df2[df2['Status'] == 'Opened']
# opened_cases_count = opened_cases_df.groupby('Assigned to').size().reset_index(name='Opened Cases')

# # Get the Top 5 Employees with the Most Opened Cases
# top_5_opened = opened_cases_count.nlargest(5, 'Opened Cases')

# # Get the Bottom 5 Employees with the Least Opened Cases
# bottom_5_opened = opened_cases_count.nsmallest(5, 'Opened Cases')

# # Display the results
# st.subheader("Top 5 Employees with the Most Opened Cases 🔓")
# st.write(top_5_opened)

# st.subheader("Bottom 5 Employees with the Least Opened Cases 🔒")
# st.write(bottom_5_opened)





# import streamlit as st
# import pandas as pd

# # Define the data
# data = {
#     "Assigned to": ["Sarah Eguaomon", "Adeyemi Akanji", "Abdul-Qudus Adekoya", "Adewale Bajela", "Adewale Bajela", "Adewale Bajela", "Adewale Bajela", "Adewale Bajela", "Adewale Bajela"],
#     "Case Number": ["CRM244032", "CRM244122", "CRM243794", "CRM244094", "CRM244116", "CRM244058", "CRM243891", "CRM243835", "CRM243810"],
#     "Customer Id": ["18672525", "21085295", "", "21848666", "22091482", "22160887", "6223131", "17111523", "7323860"],
#     "Date Created": ["01/19/2025 04:56pm", "01/19/2025 08:38pm", "01/19/2025 10:41am", "01/19/2025 07:08pm", "01/19/2025 08:22pm", "01/19/2025 05:40pm", "01/19/2025 12:59pm", "01/19/2025 11:49am", "01/19/2025 11:08am"],
#     "Date Modified": ["01/22/2025 06:19pm", "01/22/2025 04:57pm", "01/22/2025 01:50pm", "01/22/2025 11:55am", "01/22/2025 11:54am", "01/22/2025 11:54am", "01/22/2025 11:54am", "01/22/2025 11:53am", "01/22/2025 11:53am"],
#     "Lead Consumer": ["Mr. ABDULLAHI BILAL", "API Leads", "API Leads", "Mr. OTOIKHILA BLESSINGS", "Mr. STEPHEN EFENA", "Dauda Funlonsho", "Mr. CHRISTOPHER CHRISTOPHER AFOR KINGSLEY", "KELVIN AGU", "Mr. ABDULWAHAB AMOTO HASSAN"],
#     "Online Users": ["Self-Exclusion/ Self Disabled account", "", "", "", "", "Change of any account details", "Sport Bonus Complaint", "", "Mistakenly crediting another user, a deposit"],
#     "General complaint": ["", "Betslip Complaints", "", "General inquiries", "Betslip Complaints", "", "", "General inquiries", ""],
#     "Agent": ["", "", "", "", "", "", "", "", ""],
#     "Status": ["Resolved", "In Progress", "Opened", "Resolved", "Resolved", "Resolved", "Resolved", "Resolved", "Resolved"]
# }

# # Create the DataFrame
# df = pd.DataFrame(data)

# # Ensure consistency in status naming (e.g., standardizing 'In Progress' format)
# df["Status"] = df["Status"].str.title()

# # Remove missing values and exclude 'API Leads'
# df = df.dropna()
# df = df[df['Lead Consumer'] != 'API Leads']

# # Merge the 'Online Users', 'General complaint', and 'Agent' columns
# df['Merged'] = df['Online Users'].fillna('') + ' ' + df['General complaint'].fillna('') + ' ' + df['Agent'].fillna('')

# # df['Merged'] = df[['Online Users', 'General complaint', 'Agent']].apply(lambda x: ', '.join(x.dropna().astype(str)), axis=1)

# # Drop unnecessary columns
# df_cleaned = df.drop(columns=['Online Users', 'General complaint', 'Agent'])

# # Streamlit UI
# st.title("Case Management Dashboard")
# st.write("This dashboard displays the case management data and insights.")
# st.markdown("""
# This is my first interactive dashboard that explain everything about our report**

# It summarize our performaces


# """)

# # Display the dataset in a table
# st.subheader("Case Data")
# st.dataframe(df_cleaned)

# # Define all possible statuses (including missing ones)
# all_statuses = ["All", "Resolved", "In Progress", "Opened", "Pending", "Escalated"]

# # Streamlit dropdown for status filtering
# status_filter = st.selectbox('Filter by Status:', all_statuses)

# # Apply filter logic
# if status_filter == "All":
#     filtered_df = df_cleaned
# else:
#     filtered_df = df_cleaned[df_cleaned['Status'] == status_filter]

# # Display filtered results
# st.subheader(f"Filtered Cases For: {status_filter}")
# st.dataframe(filtered_df)

# # Option to download the cleaned dataset
# st.subheader("Download Cleaned Data")
# csv = df_cleaned.to_csv(index=False)

# st.download_button(
#     label="Download CSV",
#     data=csv,
#     file_name="cleaned_case_data.csv",
#     mime="text/csv"
# )







# import streamlit as st
# import pandas as pd

# # Define the data (same as before)
# data = {
#     "Assigned to": ["Sarah Eguaomon", "Adeyemi Akanji", "Abdul-Qudus Adekoya", "Adewale Bajela", "Adewale Bajela", "Adewale Bajela", "Adewale Bajela", "Adewale Bajela", "Adewale Bajela"],
#     "Case Number": ["CRM244032", "CRM244122", "CRM243794", "CRM244094", "CRM244116", "CRM244058", "CRM243891", "CRM243835", "CRM243810"],
#     "Customer Id": ["18672525", "21085295", "", "21848666", "22091482", "22160887", "6223131", "17111523", "7323860"],
#     "Date Created": ["01/19/2025 04:56pm", "01/19/2025 08:38pm", "01/19/2025 10:41am", "01/19/2025 07:08pm", "01/19/2025 08:22pm", "01/19/2025 05:40pm", "01/19/2025 12:59pm", "01/19/2025 11:49am", "01/19/2025 11:08am"],
#     "Date Modified": ["01/22/2025 06:19pm", "01/22/2025 04:57pm", "01/22/2025 01:50pm", "01/22/2025 11:55am", "01/22/2025 11:54am", "01/22/2025 11:54am", "01/22/2025 11:54am", "01/22/2025 11:53am", "01/22/2025 11:53am"],
#     "Lead Consumer": ["Mr. ABDULLAHI BILAL", "API Leads", "API Leads", "Mr. OTOIKHILA BLESSINGS", "Mr. STEPHEN EFENA", "Dauda Funlonsho", "Mr. CHRISTOPHER CHRISTOPHER AFOR KINGSLEY", "KELVIN AGU", "Mr. ABDULWAHAB AMOTO HASSAN"],
#     "Online Users": ["Self-Exclusion/ Self Disabled account", "", "", "", "", "Change of any account details", "Sport Bonus Complaint", "", "Mistakenly crediting another user, a deposit"],
#     "General complaint": ["", "Betslip Complaints", "", "General inquiries", "Betslip Complaints", "", "", "General inquiries", ""],
#     "Agent": ["", "", "", "", "", "", "", "", ""],
#     "Status": ["Resolved", "In progress", "Opened", "Resolved", "Resolved", "Resolved", "Resolved", "Resolved", "Resolved"]
# }

# # Create the DataFrame
# df = pd.DataFrame(data)


# # Merge the 'Online Users', 'General complaint', and 'Agent' columns into a single column
# df['Merged'] = df['Online Users'].fillna('') + ' ' + df['General complaint'].fillna('') + ' ' + df['Agent'].fillna('')

# # Drop the original columns if no longer needed
# df_cleaned = df.drop(columns=['Online Users', 'General complaint', 'Agent'])

# # Streamlit UI
# st.title("Case Management Dashboard")
# st.write("This dashboard displays the case management data and insights.")

# # Display the dataset in a table
# st.subheader("Case Data")
# st.dataframe(df_cleaned)

# # Option to filter by Status

# status_filter = st.selectbox('Filter by Status:', df['Status'].unique())
# filtered_df = df_cleaned[df_cleaned['Status'] == status_filter]
# st.subheader(f"Filtered Cases: {status_filter}")
# st.dataframe(filtered_df)

# # Option to download the cleaned dataset
# st.subheader("Download Cleaned Data")
# csv = df_cleaned.to_csv(index=False)
# txt = df_cleaned.to_csv(index=False, sep='\t')

# st.download_button(
#     label="Download CSV",
#     data=csv,
#     file_name="cleaned_case_data.csv",
#     mime="text/csv"
# )

# st.download_button(
#     label="Download TXT",
#     data=txt,
#     file_name="cleaned_case_data.txt",
#     mime="text/plain"
# )








# import streamlit as st
# import pandas as pd

# # Sample CCRM dataset with ticket statuses
# data = {
#     "Ticket ID": [101, 102, 103, 104, 105, 106],
#     "Customer": ["John", "Alice", "Michael", "David", "Emma", "Sophia"],
#     "Status": ["Pending", "Escalated", "Resolved", "Escalated", "Escalated", "Resolved"],
#     "Issue Type": ["Payment", "Withdrawal", "Login", "Bonus", "Deposit", "Account"],
# }
# df = pd.DataFrame(data)

# # Streamlit app
# st.title("📊 Customer Complaint Resolution (CCRM) Dashboard")

# # Dropdown for status filter
# status_filter = st.selectbox("Select Ticket Status", ["All", "Pending", "Resolved", "Escalated"])

# # Apply filtering logic
# if status_filter != "All":
#     filtered_df = df[df["Status"] == status_filter]
# else:
#     df = df  # Show all if "All" is selected

# # Display filtered data
# st.write(f"### Showing {len(df)} tickets for status: **{status_filter}**")
# st.dataframe(df)

# # KPI Metrics
# total_tickets = len(df)
# resolved_tickets = len(df[df["Status"] == "Resolved"])
# pending_tickets = len(df[df["Status"] == "Pending"])
# escalated_tickets = len(df[df["Status"] == "Escalated"])

# # Show metrics
# col1, col2, col3, col4 = st.columns(4)
# col1.metric("Total Tickets", total_tickets)
# col2.metric("Resolved Tickets", resolved_tickets, delta=f"+{resolved_tickets - pending_tickets}")
# col3.metric("Escalated Tickets", escalated_tickets, delta=f"+{escalated_tickets - pending_tickets}")
# col4.metric("Pending Tickets", pending_tickets, delta=f"+{pending_tickets - pending_tickets}")









# import streamlit as st

# st.title("Streamlit Dashboard with Emojis 🎯")
# st.subheader("Welcome to the Dashboard 🧑‍💻")
# st.write("This dashboard is all about tracking cases and performance! 💼")

# st.markdown("""
#     **Case Status:**
#     - Resolved  🌐
#     - Opened 📂
#     - Escalated 🔥
#     - In Progress 🛠️
# """)

# st.button("Download 📥")







# import streamlit as st
# import pandas as pd

# # Employee leave dataset
# data = {
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
#     'Reason': ['Annual Leave', 'Personal Leave', 'Sick Leave', 'Casual Leave',
#                'Annual Leave', 'Maternity Leave', 'Paternity Leave', 'Compassionate Leave', 'Study Leave']
# }
# df = pd.DataFrame(data)

# # Convert dates to datetime
# df['Start_Date'] = pd.to_datetime(df['Start_Date'])
# df['End_Date'] = pd.to_datetime(df['End_Date'])
# df['Month'] = df['Start_Date'].dt.month  # Extract month as a number

# # Map month numbers to names
# month_name_map = {
#     1: 'January', 2: 'February', 3: 'March', 4: 'April',
#     5: 'May', 6: 'June', 7: 'July', 8: 'August', 9: 'September',
#     10: 'October', 11: 'November', 12: 'December'
# }
# df['Month_Name'] = df['Month'].map(month_name_map)

# # Group data by month and calculate total leave taken
# monthly_trends = df.groupby('Month_Name', sort=False)['Leave_Taken'].sum().reset_index()

# # Sort by month order for correct display
# month_order = list(month_name_map.values())  # Ensure order by month names
# monthly_trends['Month_Name'] = pd.Categorical(monthly_trends['Month_Name'], categories=month_order, ordered=True)
# monthly_trends = monthly_trends.sort_values('Month_Name')

# # Determine the month with the most and least leave taken
# most_leave_month = monthly_trends.loc[monthly_trends['Leave_Taken'].idxmax()]
# least_leave_month = monthly_trends.loc[monthly_trends['Leave_Taken'].idxmin()]

# # Streamlit interface
# st.title("Employee Leave Dashboard")

# # Display monthly trends in a table format
# st.write("### Monthly Leave Trends")
# st.table(monthly_trends.rename(columns={'Month_Name': 'Month', 'Leave_Taken': 'Total Leave Taken'}))

# # Highlight the months with most and least leave taken
# st.write(f"The month with the **most leave taken** is **{most_leave_month['Month_Name']}** with **{most_leave_month['Leave_Taken']} days**.")
# st.write(f"The month with the **least leave taken** is **{least_leave_month['Month_Name']}** with **{least_leave_month['Leave_Taken']} days**.")

# # Visualize the monthly trends
# st.write("### Leave Trend Visualization")
# st.line_chart(monthly_trends.set_index('Month_Name')['Leave_Taken'])






# import streamlit as st
# import pandas as pd

# # Employee leave dataset
# data = {
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
#     'Reason': ['Annual Leave', 'Personal Leave', 'Sick Leave', 'Casual Leave',
#                'Annual Leave', 'Maternity Leave', 'Paternity Leave', 'Compassionate Leave', 'Study Leave']
# }
# df = pd.DataFrame(data)

# # Convert dates to datetime
# df['Start_Date'] = pd.to_datetime(df['Start_Date'])
# df['End_Date'] = pd.to_datetime(df['End_Date'])
# df['Month'] = df['Start_Date'].dt.month  # Extract month as a number

# # Map month numbers to names
# month_name_map = {
#     1: 'January', 2: 'February', 3: 'March', 4: 'April',
#     5: 'May', 6: 'June', 7: 'July', 8: 'August', 9: 'September',
#     10: 'October', 11: 'November', 12: 'December'
# }
# df['Month_Name'] = df['Month'].map(month_name_map)

# # Group data by month and calculate total leave taken
# monthly_trends = df.groupby('Month_Name', sort=False)['Leave_Taken'].sum().reset_index()



# # Sort by month order for correct display
# month_order = list(month_name_map.values())  # Ensure order by month names
# monthly_trends['Month_Name'] = pd.Categorical(monthly_trends['Month_Name'], categories=month_order, ordered=True)
# monthly_trends = monthly_trends.sort_values('Month_Name')

# monthly_trends2 = df.groupby('Month')['Leave_Taken'].sum().reset_index()


# with st.expander("welcome"):

#    # Streamlit App
#    st.title("Employee Leave Trends")
#    st.dataframe(df)

# st.write(monthly_trends2)


# st.header("Monthly Leave Trends")
# st.bar_chart(data=monthly_trends, x='Month_Name', y='Leave_Taken')












# import streamlit as st
# import pandas as pd
# import matplotlib.pyplot as plt

# # Set the page layout to wide
# st.set_page_config(layout='wide')

# # Sample leave dataset
# leave_data = {
#     "Employee Name": ["John Smith", "Alice Johnson", "Robert Brown", "Emily Davis", "Michael Wilson"],
#     "Leave Type": ["Annual", "Sick", "Casual", "Maternity", "Annual"],
#     "Days Taken": [10, 5, 2, 90, 15],
#     "Remaining Days": [5, 10, 8, 0, 5],
# }

# # Convert data to DataFrame
# df = pd.DataFrame(leave_data)

# # Calculate total leave taken and remaining
# total_days_taken = df["Days Taken"].sum()
# total_remaining_days = df["Remaining Days"].sum()

# # Group by Leave Type to show aggregate leave data
# leave_summary = df.groupby("Leave Type")[["Days Taken", "Remaining Days"]].sum().reset_index()

# # Streamlit app
# st.title("Leave Management Dashboard")
# st.subheader("Leave Details for Employees")

# # Display the main DataFrame
# st.dataframe(df, use_container_width=True)

# # Display totals
# st.markdown("### Totals")
# st.write(f"**Total Days Taken:** {total_days_taken}")
# st.write(f"**Total Remaining Days:** {total_remaining_days}")

# # Display the summary table
# st.markdown("### Leave Summary by Type")
# st.dataframe(leave_summary, use_container_width=True)

# # Plotting a line chart for leave type distribution
# st.markdown("### Leave Type Distribution (Line Chart)")

# # Create a figure
# fig, ax = plt.subplots(figsize=(10, 6))

# # Plot "Days Taken" and "Remaining Days" as lines
# ax.plot(leave_summary["Leave Type"], leave_summary["Days Taken"], marker='o', label="Days Taken", color='skyblue', linewidth=2)
# ax.plot(leave_summary["Leave Type"], leave_summary["Remaining Days"], marker='o', label="Remaining Days", color='orange', linewidth=2)

# # Set titles and labels
# ax.set_title('Leave Type Distribution (Days Taken vs Remaining Days)', fontsize=16, fontweight='bold')
# ax.set_xlabel('Leave Type', fontsize=12, fontweight='bold')
# ax.set_ylabel('Number of Days', fontsize=12, fontweight='bold')

# # Add a legend
# ax.legend(title='Leave Metrics', fontsize=10, title_fontsize=12)

# # Rotate x-axis labels for better readability
# ax.set_xticks(range(len(leave_summary["Leave Type"])))
# ax.set_xticklabels(leave_summary["Leave Type"], rotation=45, ha='right', fontsize=10)

# # Pass the figure to st.pyplot()
# st.pyplot(fig)

# # Add some interactivity: Filter by leave type
# st.markdown("### Filter Leave Data by Type")
# leave_type = st.selectbox("Select Leave Type:", df["Leave Type"].unique())
# filtered_data = df[df["Leave Type"] == leave_type]
# st.write(f"Employees with **{leave_type} Leave**:")
# st.dataframe(filtered_data, use_container_width=True)

















# import streamlit as st
# import pandas as pd
# import matplotlib.pyplot as plt

# # Set the page layout to wide
# st.set_page_config(layout='wide')

# # Sample leave dataset
# leave_data = {
#     "Employee Name": ["John Smith", "Alice Johnson", "Robert Brown", "Emily Davis", "Michael Wilson"],
#     "Leave Type": ["Annual", "Sick", "Casual", "Maternity", "Annual"],
#     "Days Taken": [10, 5, 2, 80, 15],
#     "Remaining Days": [5, 10, 8, 0, 5],
# }

# # Convert data to DataFrame
# df = pd.DataFrame(leave_data)

# # Calculate total leave taken and remaining
# total_days_taken = df["Days Taken"].sum()
# total_remaining_days = df["Remaining Days"].sum()

# # Group by Leave Type to show aggregate leave data
# leave_summary = df.groupby("Leave Type")[["Days Taken", "Remaining Days"]].sum().reset_index()

# # Streamlit app
# st.title("Leave Management Dashboard")
# st.subheader("Leave Details for Employees")

# # Display the main DataFrame
# st.dataframe(df, use_container_width=True)

# # Display totals
# st.markdown("### Totals")
# st.write(f"**Total Days Taken:** {total_days_taken}")
# st.write(f"**Total Remaining Days:** {total_remaining_days}")

# # Display the summary table
# st.markdown("### Leave Summary by Type")
# st.dataframe(leave_summary, use_container_width=True)


# # Add some interactivity: Filter by leave type
# st.markdown("### Filter Leave Data by Type")
# leave_type = st.selectbox("Select Leave Type:", df["Leave Type"].unique())
# filtered_data = df[df["Leave Type"] == leave_type]
# st.write(f"Employees with **{leave_type} Leave**:")
# st.dataframe(filtered_data, use_container_width=True)



# # Plotting a bar chart for leave type distribution
# st.markdown("### Leave Type Distribution")

# # Create a figure
# fig, ax = plt.subplots(figsize=(10, 6))

# # Plot both "Days Taken" and "Remaining Days"
# leave_summary.set_index('Leave Type')[['Days Taken', 'Remaining Days']].plot(
#     kind='line', ax=ax, color=['green', 'orange'], edgecolor='red'
# )

# # Set titles and labels
# ax.set_title('Leave Type Distribution (Days Taken vs Remaining Days)', fontsize=16, fontweight='bold')
# ax.set_xlabel('Leave Type', fontsize=12, fontweight='bold')
# ax.set_ylabel('Number of Days', fontsize=12, fontweight='bold')

# # Rotate x-axis labels for better readability
# ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha='right', fontsize=8)

# # Add a legend
# ax.legend(title='Leave Metrics', fontsize=10)

# # Pass the figure to st.pyplot()
# st.pyplot(fig)














# import streamlit as st
# import pandas as pd
# import matplotlib.pyplot as plt

# # Sample leave dataset
# leave_data = {
#     "Employee Name": ["John Smith", "Alice Johnson", "Robert Brown", "Emily Davis", "Michael Wilson"],
#     "Leave Type": ["Annual", "Sick", "Casual", "Maternity", "Annual"],
#     "Days Taken": [10, 5, 2, 90, 15],
#     "Remaining Days": [5, 10, 8, 0, 5],
# }

# # Convert data to DataFrame
# df = pd.DataFrame(leave_data)

# # Calculate total leave taken and remaining
# total_days_taken = df["Days Taken"].sum()
# total_remaining_days = df["Remaining Days"].sum()

# # Group by Leave Type to show aggregate leave data
# leave_summary = df.groupby("Leave Type")[["Days Taken", "Remaining Days"]].sum().reset_index()

# # Streamlit app
# st.title("Leave Management Dashboard")
# st.subheader("Leave Details for Employees")

# # Display the main DataFrame
# st.dataframe(df)

# # Display totals
# st.markdown("### Totals")
# st.write(f"**Total Days Taken:** {total_days_taken}")
# st.write(f"**Total Remaining Days:** {total_remaining_days}")

# # Display the summary table
# st.markdown("### Leave Summary by Type")
# st.dataframe(leave_summary)

# # Plotting a bar chart for leave type distribution
# st.markdown("### Leave Type Distribution")

# # Create a figure
# fig, ax = plt.subplots(figsize=(6, 4))

# # Plot both "Days Taken" and "Remaining Days"
# leave_summary.set_index('Leave Type')[['Days Taken', 'Remaining Days']].plot(
#     kind='bar', ax=ax, color=['skyblue', 'orange'], edgecolor='black'
# )

# # Set titles and labels
# ax.set_title('Leave Type Distribution (Days Taken vs Remaining Days)', fontsize=14, fontweight='bold')
# ax.set_xlabel('Leave Type', fontsize=12, fontweight='bold')
# ax.set_ylabel('Number of Days', fontsize=12, fontweight='bold')

# # Rotate x-axis labels for better readability
# ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=10)

# # Add a legend
# ax.legend(title='Leave Metrics', fontsize=10, title_fontsize=12)

# # Pass the figure to st.pyplot()
# st.pyplot(fig)

# # Add some interactivity: Filter by leave type
# st.markdown("### Filter Leave Data by Type")
# leave_type = st.selectbox("Select Leave Type:", df["Leave Type"].unique())

# if leave_type:

#      filtered_data = df[df["Leave Type"] == leave_type]
#      st.write(f"Employees with **{leave_type} Leave**:")
#      st.dataframe(filtered_data)










# CRM CODE


# import pandas as pd
# import streamlit as st
# import seaborn as sns
# import matplotlib.pyplot as plt

# # Sample dataset (replace with your actual dataset)
# data = {
#     "Assigned to": [
#         "Sarah Eguaomon", "Adeyemi Akanji", "Abdul-Qudus Adekoya", "Adewale Bajela", "Adewale Bajela",
#         "Adewale Bajela", "Adewale Bajela", "Adewale Bajela", "Adewale Bajela"
#     ],
#     "Status": ["Resolved", "In progress", "Opened", "Resolved", "Resolved", "Resolved", "Resolved", "Resolved", "Resolved"],
# }

# # Create a DataFrame
# df = pd.DataFrame(data)

# # Sidebar filter for employees
# st.sidebar.title("Filters")
# selected_status = st.sidebar.selectbox("Select Status", options=["Opened", "Resolved", "In progress", "Escalated"], index=0)

# # Filter by selected status
# filtered_df = df[df['Status'] == selected_status]

# # Group by 'Assigned to' and count cases
# status_counts = filtered_df.groupby('Assigned to').size().reset_index(name=f'{selected_status} Cases')

# # Top 5 employees with the most cases
# top_5 = status_counts.nlargest(5, f'{selected_status} Cases')

# # Bottom 5 employees with the least cases
# bottom_5 = status_counts.nsmallest(5, f'{selected_status} Cases')

# # Display top 5 and bottom 5 employees
# st.subheader(f"Top 5 Employees with Most {selected_status} Cases")
# st.table(top_5)

# st.subheader(f"Bottom 5 Employees with Least {selected_status} Cases")
# st.table(bottom_5)

# # Combine the top 5 and bottom 5 for visualization
# combined_df = pd.concat([top_5, bottom_5])

# # Bar chart for the combined data
# st.subheader(f"Bar Chart of {selected_status} Cases (Top and Bottom 5 Employees)")
# fig, ax = plt.subplots()
# sns.barplot(x='Assigned to', y=f'{selected_status} Cases', data=combined_df, ax=ax, palette='coolwarm')
# ax.set_title(f'{selected_status} Cases by Employee')
# ax.set_ylabel(f'Number of {selected_status} Cases')
# ax.set_xlabel('Assigned to')
# plt.xticks(rotation=45)
# st.pyplot(fig)







# COMPLETE CRM CODE 2



# import streamlit as st
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns

# # Filepath
# filepath = "CASES.xlsx"

# # Create the DataFrame with error handling
# try:
#     df = pd.read_excel(filepath, engine='openpyxl')  # Use 'openpyxl' for reading Excel files
# except Exception as e:
#     st.error(f"Error loading Excel file: {e}")
#     st.stop()  # Stop further execution if the file can't be loaded

# # Ensure necessary columns exist
# required_columns = ['Date Created', 'Date Modified', 'Assigned to', 'Case Number', 'Status', 'Online Users', 'General complaint', 'Agent']
# missing_columns = [col for col in required_columns if col not in df.columns]
# if missing_columns:
#     st.error(f"Missing columns: {', '.join(missing_columns)}")
#     st.stop()

# # Convert 'Date Created' and 'Date Modified' to datetime with error handling
# df['Date Created'] = pd.to_datetime(df['Date Created'], errors='coerce')  # Coerce errors if any
# df['Date Modified'] = pd.to_datetime(df['Date Modified'], errors='coerce')

# # Drop rows with NaT (invalid datetime)
# df = df.dropna(subset=['Date Created', 'Date Modified'])

# # Calculate time difference between 'Date Created' and 'Date Modified'
# df['Time Taken'] = df['Date Modified'] - df['Date Created']

# # Human-readable time difference
# df['Time Taken (Readable)'] = df['Time Taken'].apply(lambda x: f"{x.days} days, {x.seconds // 3600} hours, {(x.seconds // 60) % 60} minutes")

# # Merge issue-related columns
# df['Merged'] = df['Online Users'].fillna('') + ' ' + df['General complaint'].fillna('') + ' ' + df['Agent'].fillna('')

# # Drop unnecessary columns
# df_cleaned = df.drop(columns=['Online Users', 'General complaint', 'Agent'])

# # Streamlit UI Setup
# st.set_page_config(page_title="Case Management Dashboard", layout="wide")

# st.title("Case Management Dashboard For KCG")
# st.markdown("""
#     This dashboard provides insights into case management, helping to track cases, their statuses, and issues over time. 
#     Use the filters in the sidebar to explore case data based on 'Assigned to' and 'Status'.
# """)

# # Sidebar for Filtering
# st.sidebar.title("Filters")
# assigned_to_search = st.sidebar.selectbox('Search by Assigned to:', df['Assigned to'].unique())
# status_filter = st.sidebar.selectbox('Filter by Status:', ['All', 'Opened', 'In Progress', 'Resolved', 'Escalated'])

# # Filter dataset based on selected 'Assigned to'
# filtered_df = df_cleaned[df_cleaned['Assigned to'] == assigned_to_search]

# # Apply status filter if not "All"
# if status_filter != 'All':
#     filtered_df = filtered_df[filtered_df['Status'] == status_filter]

# # Display filtered results based on 'Assigned to' and 'Status'
# st.sidebar.subheader(f"Filtered Cases for {assigned_to_search}")
# st.sidebar.write(filtered_df[['Case Number', 'Status', 'Time Taken (Readable)']])

# # Main Content: Case Status Summary
# st.subheader("Case Status Summary")
# case_status_summary = filtered_df['Status'].value_counts()

# # Display total counts for each case status for the selected "Assigned to"
# total_cases = {
#     "Escalated": case_status_summary.get("Escalated", 0),
#     "Resolved": case_status_summary.get("Resolved", 0),
#     "In Progress": case_status_summary.get("In Progress", 0),
#     "Opened": case_status_summary.get("Opened", 0)
# }

# st.write(f"**Total Escalated Cases**: {total_cases['Escalated']}")
# st.write(f"**Total Resolved Cases**: {total_cases['Resolved']}")
# st.write(f"**Total In Progress Cases**: {total_cases['In Progress']}")
# st.write(f"**Total Opened Cases**: {total_cases['Opened']}")

# # Identify top and bottom 5 employees by 'Opened' cases
# opened_cases = df_cleaned[df_cleaned['Status'] == 'Opened']
# opened_counts = opened_cases.groupby('Assigned to').size().reset_index(name='Opened Cases')

# top_5_opened = opened_counts.nlargest(5, 'Opened Cases')
# bottom_5_opened = opened_counts.nsmallest(5, 'Opened Cases')

# # Display top and bottom 5
# st.subheader("Top 5 Employees with Most Opened Cases")
# st.table(top_5_opened)

# st.subheader("Bottom 5 Employees with Least Opened Cases")
# st.table(bottom_5_opened)

# # Top 5 Employees with Most Resolved Cases
# st.subheader("Top 5 Employees with Most Resolved Cases")
# resolved_cases = df_cleaned[df_cleaned['Status'] == 'Resolved']  # Filter for resolved cases
# resolved_count = resolved_cases.groupby('Assigned to').size().reset_index(name='Resolved Count')
# top_5_employees = resolved_count.sort_values(by='Resolved Count', ascending=False).head(5)

# st.write(top_5_employees)

# # Bottom 5 Employees with Least Resolved Cases
# st.subheader("Bottom 5 Employees with Least Resolved Cases")
# bottom_5_employees = resolved_count.sort_values(by='Resolved Count', ascending=True).head(5)

# st.write(bottom_5_employees)



# # Bar chart for top and bottom 5
# combined_opened = pd.concat([top_5_opened, bottom_5_opened])
# st.subheader("Bar Chart: Top and Bottom 5 Employees with Opened Cases")
# fig, ax = plt.subplots()
# sns.barplot(x='Assigned to', y='Opened Cases', data=combined_opened, ax=ax, palette='coolwarm')
# ax.set_title('Top and Bottom 5 Employees with Opened Cases')
# plt.xticks(rotation=45)
# st.pyplot(fig)

# # Main Content: Issue Count Over Time
# st.subheader("Issue Count Over Time")
# df_cleaned_non_na = df_cleaned[df_cleaned['Merged'].notna() & (df_cleaned['Merged'] != '')]
# issue_count_by_date = df_cleaned_non_na.groupby(df['Date Created'].dt.date)['Merged'].count()

# fig2, ax2 = plt.subplots(figsize=(10, 6))
# issue_count_by_date.plot(kind='line', ax=ax2, color='blue', marker='o')
# ax2.set_title('Issue Count Over Time')
# ax2.set_xlabel('Date')
# ax2.set_ylabel('Issue Count')
# st.pyplot(fig2)

# # Section for Data Download
# st.subheader("Download Cleaned Data")
# csv = df_cleaned.to_csv(index=False)

# st.download_button(
#     label="Download CSV",
#     data=csv,
#     file_name="cleaned_case_data.csv",
#     mime="text/csv"
# )









# COMPLETE CRM CODE 


# import streamlit as st
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns

# # Filepath
# filepath = "CASES.xlsx"

# # Create the DataFrame with error handling
# try:
#     df = pd.read_excel(filepath, engine='openpyxl')  # Use 'openpyxl' for reading Excel files
# except Exception as e:
#     st.error(f"Error loading Excel file: {e}")
#     st.stop()  # Stop further execution if the file can't be loaded

# # Ensure necessary columns exist
# required_columns = ['Date Created', 'Date Modified', 'Assigned to', 'Case Number', 'Status', 'Online Users', 'General complaint', 'Agent']
# missing_columns = [col for col in required_columns if col not in df.columns]
# if missing_columns:
#     st.error(f"Missing columns: {', '.join(missing_columns)}")
#     st.stop()

# # Convert 'Date Created' and 'Date Modified' to datetime with error handling
# df['Date Created'] = pd.to_datetime(df['Date Created'], errors='coerce')  # Coerce errors if any
# df['Date Modified'] = pd.to_datetime(df['Date Modified'], errors='coerce')

# # Drop rows with NaT (invalid datetime)
# df = df.dropna(subset=['Date Created', 'Date Modified'])

# # Calculate time difference between 'Date Created' and 'Date Modified'
# df['Time Taken'] = df['Date Modified'] - df['Date Created']

# # Human-readable time difference
# df['Time Taken (Readable)'] = df['Time Taken'].apply(lambda x: f"{x.days} days, {x.seconds // 3600} hours, {(x.seconds // 60) % 60} minutes")

# # Merge issue-related columns
# df['Merged'] = df['Online Users'].fillna('') + ' ' + df['General complaint'].fillna('') + ' ' + df['Agent'].fillna('')

# # Drop unnecessary columns
# df_cleaned = df.drop(columns=['Online Users', 'General complaint', 'Agent'])

# # Streamlit UI Setup
# st.set_page_config(page_title="Case Management Dashboard", layout="wide")

# # st.image('Bet9ja_logo.jpeg', width=300, caption='our logo')
# st.title("Case Management Dashboard For KCG")
# st.markdown("""
#     This dashboard provides insights into case management, helping to track cases, their statuses, and issues over time. 
#     Use the filters in the sidebar to explore case data based on 'Assigned to' and 'Status'.
# """)

# # Sidebar for Filtering
# st.sidebar.title("Filters")
# assigned_to_search = st.sidebar.selectbox('Search by Assigned to:', df['Assigned to'].unique())
# status_filter = st.sidebar.selectbox('Filter by Status:', ['All', 'Opened', 'In Progress', 'Resolved', 'Escalated'])

# # Filter dataset based on selected 'Assigned to'
# filtered_df = df_cleaned[df_cleaned['Assigned to'] == assigned_to_search]

# # Apply status filter if not "All"
# if status_filter != 'All':
#     filtered_df = filtered_df[filtered_df['Status'] == status_filter]

# # Display filtered results based on 'Assigned to' and 'Status'
# st.sidebar.subheader(f"Filtered Cases for {assigned_to_search}")
# st.sidebar.write(filtered_df[['Case Number', 'Status', 'Time Taken (Readable)']])

# # Main Content: Case Status Summary
# st.subheader("Case Status Summary")
# case_status_summary = filtered_df['Status'].value_counts()

# # Display total counts for each case status for the selected "Assigned to"
# total_cases = {
#     "Escalated": case_status_summary.get("Escalated", 0),
#     "Resolved": case_status_summary.get("Resolved", 0),
#     "In Progress": case_status_summary.get("In Progress", 0),
#     "Opened": case_status_summary.get("Opened", 0)
# }

# st.write(f"**Total Escalated Cases**: {total_cases['Escalated']}")
# st.write(f"**Total Resolved Cases**: {total_cases['Resolved']}")
# st.write(f"**Total In Progress Cases**: {total_cases['In Progress']}")
# st.write(f"**Total Opened Cases**: {total_cases['Opened']}")

# # Main Content: Case Status Distribution
# st.subheader("Case Status Distribution")
# fig, ax = plt.subplots(figsize=(10, 6))
# sns.barplot(x=case_status_summary.index, y=case_status_summary.values, ax=ax, palette="viridis")
# ax.set_title('Distribution of Case Statuses')
# ax.set_ylabel('Number of Cases')
# st.pyplot(fig)

# # Main Content: Count of Each Status by Assigned to
# st.subheader("Count of Each Status by Assigned to")
# status_by_assigned = filtered_df.groupby(['Assigned to', 'Status']).size().reset_index(name='Count')
# st.write(status_by_assigned)

# # Main Content: Issue Count Over Time
# st.subheader("Issue Count Over Time")
# df_cleaned_non_na = df_cleaned[df_cleaned['Merged'].notna() & (df_cleaned['Merged'] != '')]
# issue_count_by_date = df_cleaned_non_na.groupby(df['Date Created'].dt.date)['Merged'].count()

# fig2, ax2 = plt.subplots(figsize=(10, 6))
# issue_count_by_date.plot(kind='line', ax=ax2, color='blue', marker='o')
# ax2.set_title('Issue Count Over Time')
# ax2.set_xlabel('Date')
# ax2.set_ylabel('Issue Count')
# st.pyplot(fig2)

# # Display the total count of each issue
# st.subheader("Total Count of Each Issue")
# issue_counts = df_cleaned_non_na['Merged'].value_counts()
# st.write(issue_counts)

# # Section for Data Download
# st.subheader("Download Cleaned Data")
# csv = df_cleaned.to_csv(index=False)
# txt = df_cleaned.to_csv(index=False, sep='\t')

# st.download_button(
#     label="Download CSV",
#     data=csv,
#     file_name="cleaned_case_data.csv",
#     mime="text/csv"
# )


# st.download_button(
#     label="Download TXT",
#     data=csv,
#     file_name="cleaned_case_data.csv",
#     mime="text/csv"
# )




# COMPLETE CRM CODE 3 GOOD ONE




# import streamlit as st
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns

# # Filepath
# filepath = "CASES.xlsx"

# # Create the DataFrame with error handling
# try:
#     df = pd.read_excel(filepath, engine='openpyxl')  # Use 'openpyxl' for reading Excel files
# except Exception as e:
#     st.error(f"Error loading Excel file: {e}")
#     st.stop()  # Stop further execution if the file can't be loaded

# # Ensure necessary columns exist
# required_columns = ['Date Created', 'Date Modified', 'Assigned to', 'Case Number', 'Status', 'Online Users', 'General complaint', 'Agent']
# missing_columns = [col for col in required_columns if col not in df.columns]
# if missing_columns:
#     st.error(f"Missing columns: {', '.join(missing_columns)}")
#     st.stop()

# # Convert 'Date Created' and 'Date Modified' to datetime with error handling
# df['Date Created'] = pd.to_datetime(df['Date Created'], errors='coerce')  # Coerce errors if any
# df['Date Modified'] = pd.to_datetime(df['Date Modified'], errors='coerce')

# # Drop rows with NaT (invalid datetime)
# df = df.dropna(subset=['Date Created', 'Date Modified'])

# # Calculate time difference between 'Date Created' and 'Date Modified'
# df['Time Taken'] = df['Date Modified'] - df['Date Created']

# # Human-readable time difference
# df['Time Taken (Readable)'] = df['Time Taken'].apply(lambda x: f"{x.days} days, {x.seconds // 3600} hours, {(x.seconds // 60) % 60} minutes")

# # Merge issue-related columns
# df['Merged'] = df['Online Users'].fillna('') + ' ' + df['General complaint'].fillna('') + ' ' + df['Agent'].fillna('')

# # Drop unnecessary columns
# df_cleaned = df.drop(columns=['Online Users', 'General complaint', 'Agent'])

# # Streamlit UI Setup
# st.set_page_config(page_title="Case Management Dashboard", layout="wide")

# # st.image('Bet9ja_logo.jpeg', width=300, caption='our logo')
# st.title("Case Management Dashboard For KCG")
# st.markdown("""
#     This dashboard provides insights into case management, helping to track cases, their statuses, and issues over time. 
#     Use the filters in the sidebar to explore case data based on 'Assigned to' and 'Status'.
# """)

# # Sidebar for Filtering
# st.sidebar.title("Filters")
# assigned_to_search = st.sidebar.selectbox('Search by Assigned to:', df['Assigned to'].unique())
# status_filter = st.sidebar.selectbox('Filter by Status:', ['All', 'Opened', 'In Progress', 'Resolved', 'Escalated'])

# # Filter dataset based on selected 'Assigned to'
# filtered_df = df_cleaned[df_cleaned['Assigned to'] == assigned_to_search]

# # Apply status filter if not "All"
# if status_filter != 'All':
#     filtered_df = filtered_df[filtered_df['Status'] == status_filter]

# # Display filtered results based on 'Assigned to' and 'Status'
# st.sidebar.subheader(f"Filtered Cases for {assigned_to_search}")
# st.sidebar.write(filtered_df[['Case Number', 'Status', 'Time Taken (Readable)']])

# # Main Content: Case Status Summary
# st.subheader("Case Status Summary")
# case_status_summary = filtered_df['Status'].value_counts()

# # Display total counts for each case status for the selected "Assigned to"
# total_cases = {
#     "Escalated": case_status_summary.get("Escalated", 0),
#     "Resolved": case_status_summary.get("Resolved", 0),
#     "In Progress": case_status_summary.get("In Progress", 0),
#     "Opened": case_status_summary.get("Opened", 0)
# }

# st.write(f"**Total Escalated Cases**: {total_cases['Escalated']}")
# st.write(f"**Total Resolved Cases**: {total_cases['Resolved']}")
# st.write(f"**Total In Progress Cases**: {total_cases['In Progress']}")
# st.write(f"**Total Opened Cases**: {total_cases['Opened']}")

# # Main Content: Case Status Distribution
# st.subheader("Case Status Distribution")
# fig, ax = plt.subplots(figsize=(10, 6))
# sns.barplot(x=case_status_summary.index, y=case_status_summary.values, ax=ax, palette="viridis")
# ax.set_title('Distribution of Case Statuses')
# ax.set_ylabel('Number of Cases')
# st.pyplot(fig)

# # Main Content: Count of Each Status by Assigned to
# st.subheader("Count of Each Status by Assigned to")
# status_by_assigned = filtered_df.groupby(['Assigned to', 'Status']).size().reset_index(name='Count')
# st.write(status_by_assigned)

# # Main Content: Issue Count Over Time
# st.subheader("Issue Count Over Time")
# df_cleaned_non_na = df_cleaned[df_cleaned['Merged'].notna() & (df_cleaned['Merged'] != '')]
# issue_count_by_date = df_cleaned_non_na.groupby(df['Date Created'].dt.date)['Merged'].count()

# fig2, ax2 = plt.subplots(figsize=(10, 6))
# issue_count_by_date.plot(kind='line', ax=ax2, color='blue', marker='o')
# ax2.set_title('Issue Count Over Time')
# ax2.set_xlabel('Date')
# ax2.set_ylabel('Issue Count')
# st.pyplot(fig2)

# # Display the total count of each issue
# st.subheader("Total Count of Each Issue")
# issue_counts = df_cleaned_non_na['Merged'].value_counts()
# st.write(issue_counts)


# # Section to get Top 5 Employees with Most Resolved Cases and Least Resolved Cases
# resolved_cases_df = df_cleaned[df_cleaned['Status'] == 'Resolved']
# resolved_cases_count = resolved_cases_df.groupby('Assigned to').size().reset_index(name='Resolved Cases')

# # Get the Top 5 Employees with the Most Resolved Cases
# top_5_resolved = resolved_cases_count.nlargest(5, 'Resolved Cases')

# # Get the Bottom 5 Employees with the Least Resolved Cases
# bottom_5_resolved = resolved_cases_count.nsmallest(5, 'Resolved Cases')

# # Display the results
# st.subheader("Top 5 Employees with the Most Resolved Cases")
# st.write(top_5_resolved)

# st.subheader("Bottom 5 Employees with the Least Resolved Cases")
# st.write(bottom_5_resolved)

# # Section to get Top 5 Employees with Most Opened Cases and Bottom 5 Employees with Least Opened Cases
# opened_cases_df = df_cleaned[df_cleaned['Status'] == 'Opened']
# opened_cases_count = opened_cases_df.groupby('Assigned to').size().reset_index(name='Opened Cases')

# # Get the Top 5 Employees with the Most Opened Cases
# top_5_opened = opened_cases_count.nlargest(5, 'Opened Cases')

# # Get the Bottom 5 Employees with the Least Opened Cases
# bottom_5_opened = opened_cases_count.nsmallest(5, 'Opened Cases')

# # Display the results
# st.subheader("Top 5 Employees with the Most Opened Cases")
# st.write(top_5_opened)

# st.subheader("Bottom 5 Employees with the Least Opened Cases")
# st.write(bottom_5_opened)



# # Section for Data Download
# st.subheader("Download Cleaned Data")
# csv = df_cleaned.to_csv(index=False)
# txt = df_cleaned.to_csv(index=False, sep='\t')

# st.download_button(
#     label="Download CSV",
#     data=csv,
#     file_name="cleaned_case_data.csv",
#     mime="text/csv"
# )

# st.download_button(
#     label="Download TXT",
#     data=csv,
#     file_name="cleaned_case_data.csv",
#     mime="text/csv"
# )







# CRM WITH DONOT AND OTHER FEATURES


# import streamlit as st
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns

# # Filepath
# filepath = "CASES.xlsx"

# # Create the DataFrame with error handling
# try:
#     df = pd.read_excel(filepath, engine='openpyxl')  # Use 'openpyxl' for reading Excel files
# except Exception as e:
#     st.error(f"Error loading Excel file: {e}")
#     st.stop()  # Stop further execution if the file can't be loaded

# # Ensure necessary columns exist
# required_columns = ['Date Created', 'Date Modified', 'Assigned to', 'Case Number', 'Status', 'Online Users', 'General complaint', 'Agent']
# missing_columns = [col for col in required_columns if col not in df.columns]
# if missing_columns:
#     st.error(f"Missing columns: {', '.join(missing_columns)}")
#     st.stop()

# # Convert 'Date Created' and 'Date Modified' to datetime with error handling
# df['Date Created'] = pd.to_datetime(df['Date Created'], errors='coerce')  # Coerce errors if any
# df['Date Modified'] = pd.to_datetime(df['Date Modified'], errors='coerce')

# # Drop rows with NaT (invalid datetime)
# df = df.dropna(subset=['Date Created', 'Date Modified'])

# # Calculate time difference between 'Date Created' and 'Date Modified'
# df['Time Taken'] = df['Date Modified'] - df['Date Created']

# # Human-readable time difference
# df['Time Taken (Readable)'] = df['Time Taken'].apply(lambda x: f"{x.days} days, {x.seconds // 3600} hours, {(x.seconds // 60) % 60} minutes")

# # Merge issue-related columns
# df['Merged'] = df['Online Users'].fillna('') + ' ' + df['General complaint'].fillna('') + ' ' + df['Agent'].fillna('')

# # Drop unnecessary columns
# df_cleaned = df.drop(columns=['Online Users', 'General complaint', 'Agent'])

# # Streamlit UI Setup
# st.set_page_config(page_title="Case Management Dashboard", layout="wide", page_icon="📊")

# # Add the Streamlit logo and some emojis to enhance the UI
# st.image('https://streamlit.io/images/brand/streamlit-mark-color.svg', width=150, caption='Streamlit Dashboard 🎯')
# st.title("Case Management Dashboard For KCG 🧑‍💻")
# st.markdown("""
#     Welcome to the **Case Management Dashboard**! 📊
#     This dashboard helps track cases, statuses, and issues efficiently. Use the filters to explore detailed case data based on the selected categories. 🕵️‍♂️
# """)

# # Sidebar for Filtering
# st.sidebar.title("Filters 🔍")

# # Assigned to filter
# assigned_to_search = st.sidebar.selectbox('Search by Assigned to:', ['All'] + df['Assigned to'].unique().tolist())
# # Status filter
# status_filter = st.sidebar.selectbox('Filter by Status:', ['All'] + df['Status'].unique().tolist())

# # Date filter
# st.sidebar.subheader("Date Filters 📅")
# start_date = st.sidebar.date_input("Start Date", df['Date Created'].min())
# end_date = st.sidebar.date_input("End Date", df['Date Created'].max())

# # Apply filters
# filtered_df = df_cleaned.copy()

# if assigned_to_search != 'All':
#     filtered_df = filtered_df[filtered_df['Assigned to'] == assigned_to_search]
# if status_filter != 'All':
#     filtered_df = filtered_df[filtered_df['Status'] == status_filter]
# filtered_df = filtered_df[(filtered_df['Date Created'] >= pd.Timestamp(start_date)) & 
#                           (filtered_df['Date Created'] <= pd.Timestamp(end_date))]

# # Display filtered results in the sidebar
# st.sidebar.subheader(f"Filtered Cases for {assigned_to_search}")
# st.sidebar.write(filtered_df[['Case Number', 'Status', 'Time Taken (Readable)']])

# # Metrics for quick insights
# st.subheader("Key Metrics 📊")

# # Calculate average time taken and convert to a readable format
# if len(filtered_df) > 0:
#     avg_time_taken = filtered_df['Time Taken'].mean()  # This is a Timedelta object
#     avg_time_str = f"{avg_time_taken.days} days, {avg_time_taken.seconds // 3600} hours, {(avg_time_taken.seconds // 60) % 60} minutes"
# else:
#     avg_time_str = "N/A"

# st.metric("Total Cases", len(filtered_df))
# st.metric("Average Time Taken", avg_time_str)  # Display human-readable average time
# st.metric("Escalated Cases", len(filtered_df[filtered_df['Status'] == "Escalated"]))

# # Main Content: Case Status Summary
# st.subheader("Case Status Summary 📝")
# case_status_summary = filtered_df['Status'].value_counts()

# # Display total counts for each case status for the selected filters
# for status, count in case_status_summary.items():
#     st.write(f"**Total {status} Cases**: {count} 🔹")

# # Main Content: Case Status Distribution with Donut Chart
# st.subheader("Case Status Distribution 📊")

# fig, ax = plt.subplots(figsize=(8, 6))
# wedges, texts, autotexts = ax.pie(
#     case_status_summary.values, labels=case_status_summary.index, autopct='%1.1f%%', startangle=90, pctdistance=0.85, colors=sns.color_palette("pastel")
# )
# # Adding the donut hole
# centre_circle = plt.Circle((0, 0), 0.70, fc='white')
# fig.gca().add_artist(centre_circle)

# ax.set_title("Case Status Distribution (Donut Chart)")
# st.pyplot(fig)

# # Main Content: Issue Count Over Time
# st.subheader("Issue Count Over Time ⏳")
# df_cleaned_non_na = df_cleaned[df_cleaned['Merged'].notna() & (df_cleaned['Merged'] != '')]
# issue_count_by_date = df_cleaned_non_na.groupby(df['Date Created'].dt.date)['Merged'].count()

# fig2, ax2 = plt.subplots(figsize=(10, 6))
# issue_count_by_date.plot(kind='line', ax=ax2, color='blue', marker='o')
# ax2.set_title('Issue Count Over Time')
# ax2.set_xlabel('Date')
# ax2.set_ylabel('Issue Count')
# st.pyplot(fig2)



# # Section to get Top 5 Employees with Most Resolved Cases and Least Resolved Cases
# resolved_cases_df = df_cleaned[df_cleaned['Status'] == 'Resolved']
# resolved_cases_count = resolved_cases_df.groupby('Assigned to').size().reset_index(name='Resolved Cases')

# # Get the Top 5 Employees with the Most Resolved Cases
# top_5_resolved = resolved_cases_count.nlargest(5, 'Resolved Cases')

# # Get the Bottom 5 Employees with the Least Resolved Cases
# bottom_5_resolved = resolved_cases_count.nsmallest(5, 'Resolved Cases')

# # Display the results
# st.subheader("Top 5 Employees with the Most Resolved Cases 🏆")
# st.write(top_5_resolved)

# st.subheader("Bottom 5 Employees with the Least Resolved Cases 👎")
# st.write(bottom_5_resolved)


# # Issue Count Section
# st.subheader("Total Count of Each Issue 🔢")

# # Count the unique issues in the 'Merged' column
# if not df_cleaned['Merged'].isna().all():
#     issue_counts = df_cleaned['Merged'].value_counts()  # Count occurrences of each issue

#     # Display the issue counts as a DataFrame for clarity
#     issue_counts_df = issue_counts.reset_index()
#     issue_counts_df.columns = ['Issue Description', 'Number of Cases']
#     st.write(issue_counts_df)

#     # Add a bar chart to visually represent issue counts
#     st.subheader("Issue Count Distribution 📊")
#     fig, ax = plt.subplots(figsize=(12, 6))
#     sns.barplot(data=issue_counts_df.head(10), x='Number of Cases', y='Issue Description', palette='magma', ax=ax)
#     ax.set_title('Top 10 Most Frequent Issues')
#     ax.set_xlabel('Number of Cases')
#     ax.set_ylabel('Issue Description')
#     st.pyplot(fig)
# else:
#     st.write("No issues found in the dataset.")


# # Section for Data Download
# st.subheader("Download Cleaned Data 📥")
# csv = df_cleaned.to_csv(index=False)
# txt = df_cleaned.to_csv(index=False, sep='\t')

# st.download_button(
#     label="Download CSV 📂",
#     data=csv,
#     file_name="cleaned_case_data.csv",
#     mime="text/csv"
# )

# st.download_button(
#     label="Download TXT 📑",
#     data=txt,
#     file_name="cleaned_case_data.txt",
#     mime="text/plain"
# )












# CRM WITH DONURT


# import streamlit as st
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns

# # Filepath
# filepath = "CASES.xlsx"

# # Create the DataFrame with error handling
# try:
#     df = pd.read_excel(filepath, engine='openpyxl')  # Use 'openpyxl' for reading Excel files
# except Exception as e:
#     st.error(f"Error loading Excel file: {e}")
#     st.stop()  # Stop further execution if the file can't be loaded

# # Ensure necessary columns exist
# required_columns = ['Date Created', 'Date Modified', 'Assigned to', 'Case Number', 'Status', 'Online Users', 'General complaint', 'Agent']
# missing_columns = [col for col in required_columns if col not in df.columns]
# if missing_columns:
#     st.error(f"Missing columns: {', '.join(missing_columns)}")
#     st.stop()

# # Convert 'Date Created' and 'Date Modified' to datetime with error handling
# df['Date Created'] = pd.to_datetime(df['Date Created'], errors='coerce')  # Coerce errors if any
# df['Date Modified'] = pd.to_datetime(df['Date Modified'], errors='coerce')

# # Drop rows with NaT (invalid datetime)
# df = df.dropna(subset=['Date Created', 'Date Modified'])

# # Calculate time difference between 'Date Created' and 'Date Modified'
# df['Time Taken'] = df['Date Modified'] - df['Date Created']

# # Human-readable time difference
# df['Time Taken (Readable)'] = df['Time Taken'].apply(lambda x: f"{x.days} days, {x.seconds // 3600} hours, {(x.seconds // 60) % 60} minutes")

# # Merge issue-related columns
# df['Merged'] = df['Online Users'].fillna('') + ' ' + df['General complaint'].fillna('') + ' ' + df['Agent'].fillna('')

# # Drop unnecessary columns
# df_cleaned = df.drop(columns=['Online Users', 'General complaint', 'Agent'])

# # Streamlit UI Setup
# st.set_page_config(page_title="Case Management Dashboard", layout="wide", page_icon="📊")

# st.title("Case Management Dashboard For KCG 🧑‍💻")
# st.markdown("""
#     Welcome to the **Case Management Dashboard**! 📊
#     Use the metrics below and visualizations to monitor the performance and distribution of cases.
# """)

# # Calculate total counts for metrics
# total_cases = len(df_cleaned)
# resolved_cases = len(df_cleaned[df_cleaned['Status'] == 'Resolved'])
# opened_cases = len(df_cleaned[df_cleaned['Status'] == 'Opened'])
# in_progress_cases = len(df_cleaned[df_cleaned['Status'] == 'In Progress'])
# escalated_cases = len(df_cleaned[df_cleaned['Status'] == 'Escalated'])

# # Display metrics
# col1, col2, col3, col4, col5 = st.columns(5)
# col1.metric("Total Cases", total_cases)
# col2.metric("Resolved Cases ✅", resolved_cases)
# col3.metric("Opened Cases 📂", opened_cases)
# col4.metric("In Progress Cases 🛠️", in_progress_cases)
# col5.metric("Escalated Cases 🔥", escalated_cases)

# # Donut chart for case distribution
# st.subheader("Case Status Distribution (Donut Chart) 🍩")
# case_status_counts = df_cleaned['Status'].value_counts()

# fig, ax = plt.subplots(figsize=(8, 6))
# ax.pie(case_status_counts, labels=case_status_counts.index, autopct='%1.1f%%', startangle=90, wedgeprops=dict(width=0.4), colors=sns.color_palette("viridis"))
# ax.set_title("Distribution of Case Statuses")
# st.pyplot(fig)

# # Main Content: Top and Bottom Employees
# st.subheader("Top and Bottom Performing Employees")

# # Get the Top 5 Employees with the Most Resolved Cases
# resolved_cases_df = df_cleaned[df_cleaned['Status'] == 'Resolved']
# resolved_cases_count = resolved_cases_df.groupby('Assigned to').size().reset_index(name='Resolved Cases')
# top_5_resolved = resolved_cases_count.nlargest(5, 'Resolved Cases')
# bottom_5_resolved = resolved_cases_count.nsmallest(5, 'Resolved Cases')

# # Display results
# st.write("**Top 5 Employees with the Most Resolved Cases 🏆**")
# st.table(top_5_resolved)

# st.write("**Bottom 5 Employees with the Least Resolved Cases 👎**")
# st.table(bottom_5_resolved)

# # Get the Top 5 Employees with the Most Opened Cases
# opened_cases_df = df_cleaned[df_cleaned['Status'] == 'Opened']
# opened_cases_count = opened_cases_df.groupby('Assigned to').size().reset_index(name='Opened Cases')
# top_5_opened = opened_cases_count.nlargest(5, 'Opened Cases')
# bottom_5_opened = opened_cases_count.nsmallest(5, 'Opened Cases')

# # Display results
# st.write("**Top 5 Employees with the Most Opened Cases 🔓**")
# st.table(top_5_opened)

# st.write("**Bottom 5 Employees with the Least Opened Cases 🔒**")
# st.table(bottom_5_opened)

# # Section for Data Download
# st.subheader("Download Cleaned Data 📥")
# csv = df_cleaned.to_csv(index=False)
# st.download_button(
#     label="Download CSV 📂",
#     data=csv,
#     file_name="cleaned_case_data.csv",
#     mime="text/csv"
# )











# # Section for Data Download
# st.subheader("Download Cleaned Data 📥")
# csv = df_cleaned.to_csv(index=False)
# txt = df_cleaned.to_csv(index=False, sep='\t')

# st.download_button(
#     label="Download CSV 📂",
#     data=csv,
#     file_name="cleaned_case_data.csv",
#     mime="text/csv"
# )

# st.download_button(
#     label="Download TXT 📑",
#     data=csv,
#     file_name="cleaned_case_data.csv",
#     mime="text/csv"
# )










# st.download_button(
#     label="Download Tab-Separated TXT",
#     data=txt,
#     file_name="cleaned_case_data.txt",
#     mime="text/plain"
# )






























# import streamlit as st
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns

# # Filepath
# filepath = "CASES.xlsx"

# # Create the DataFrame with error handling
# try:
#    df = pd.read_excel(filepath, engine='openpyxl')  # Use 'openpyxl' for reading Excel files
# except Exception as e:
#     st.error(f"Error loading Excel file: {e}")
#     st.stop()  # Stop further execution if the file can't be loaded

# # Ensure necessary columns exist
# required_columns = ['Date Created', 'Date Modified', 'Assigned to', 'Case Number', 'Status', 'Online Users', 'General complaint', 'Agent']
# missing_columns = [col for col in required_columns if col not in df.columns]
# if missing_columns:
#     st.error(f"Missing columns: {', '.join(missing_columns)}")
#     st.stop()

# # Convert 'Date Created' and 'Date Modified' to datetime with error handling
# df['Date Created'] = pd.to_datetime(df['Date Created'], errors='coerce')  # Coerce errors if any
# df['Date Modified'] = pd.to_datetime(df['Date Modified'], errors='coerce')

# # Drop rows with NaT (invalid datetime)
# df = df.dropna(subset=['Date Created', 'Date Modified'])

# # Calculate time difference between 'Date Created' and 'Date Modified'
# df['Time Taken'] = df['Date Modified'] - df['Date Created']

# # Human-readable time difference
# df['Time Taken (Readable)'] = df['Time Taken'].apply(lambda x: f"{x.days} days, {x.seconds // 3600} hours, {(x.seconds // 60) % 60} minutes")

# # Merge issue-related columns
# df['Merged'] = df['Online Users'].fillna('') + ' ' + df['General complaint'].fillna('') + ' ' + df['Agent'].fillna('')

# # Drop unnecessary columns
# df_cleaned = df.drop(columns=['Online Users', 'General complaint', 'Agent'])

# # Streamlit UI Setup
# st.set_page_config(page_title="Case Management Dashboard", layout="wide")
# st.title("Case Management Dashboard")
# st.markdown("""
#     This dashboard provides insights into case management, helping to track cases, their statuses, and issues over time. 
#     Use the filters in the sidebar to explore case data based on 'Assigned to' and 'Status'.
# """)

# # Sidebar for Filtering
# st.sidebar.title("Filters")
# assigned_to_search = st.sidebar.selectbox('Search by Assigned to:', df['Assigned to'].unique())
# status_filter = st.sidebar.selectbox('Filter by Status:', ['All', 'Opened', 'In progress', 'Resolved', 'Escalated'])

# # Filter dataset based on selected 'Assigned to'
# filtered_df = df_cleaned[df_cleaned['Assigned to'] == assigned_to_search]

# # Filter by Status if selected
# if status_filter != 'All':
#     filtered_status_df = filtered_df[filtered_df['Status'] == status_filter]
# else:
#     filtered_status_df = filtered_df

# # Display filtered results
# st.sidebar.subheader(f"All Cases for {assigned_to_search}")
# st.sidebar.write(filtered_df[['Case Number', 'Status', 'Time Taken (Readable)']])

# # Main Content: Case Status Summary
# st.subheader("Case Status Summary")
# case_status_summary = df['Status'].value_counts()

# # Display total counts for each case status
# total_cases = {
#     "Escalated": case_status_summary.get("Escalated", 0),
#     "Resolved": case_status_summary.get("Resolved", 0),
#     "In Progress": case_status_summary.get("In Progress", 0),
#     "Opened": case_status_summary.get("Opened", 0)
# }

# st.write(f"**Total Escalated Cases**: {total_cases['Escalated']}")
# st.write(f"**Total Resolved Cases**: {total_cases['Resolved']}")
# st.write(f"**Total In Progress Cases**: {total_cases['In Progress']}")
# st.write(f"**Total Opened Cases**: {total_cases['Opened']}")

# # Main Content: Case Status Distribution
# st.subheader("Case Status Distribution")
# fig, ax = plt.subplots(figsize=(10, 6))
# sns.barplot(x=case_status_summary.index, y=case_status_summary.values, ax=ax, palette="viridis")
# ax.set_title('Distribution of Case Statuses')
# ax.set_ylabel('Number of Cases')
# st.pyplot(fig)

# # Main Content: Count of Each Status by Assigned to
# st.subheader("Count of Each Status by Assigned to")
# status_by_assigned = df.groupby(['Assigned to', 'Status']).size().reset_index(name='Count')
# st.write(status_by_assigned)

# # Main Content: Issue Count Over Time
# st.subheader("Issue Count Over Time")
# df_cleaned_non_na = df_cleaned[df_cleaned['Merged'].notna() & (df_cleaned['Merged'] != '')]
# issue_count_by_date = df_cleaned_non_na.groupby(df['Date Created'].dt.date)['Merged'].count()

# fig2, ax2 = plt.subplots(figsize=(10, 6))
# issue_count_by_date.plot(kind='line', ax=ax2, color='blue', marker='o')
# ax2.set_title('Issue Count Over Time')
# ax2.set_xlabel('Date')
# ax2.set_ylabel('Issue Count')
# st.pyplot(fig2)

# # Display the total count of each issue
# st.subheader("Total Count of Each Issue")
# issue_counts = df_cleaned_non_na['Merged'].value_counts()
# st.write(issue_counts)

# # Section for Data Download
# st.subheader("Download Cleaned Data")
# csv = df_cleaned.to_csv(index=False)
# txt = df_cleaned.to_csv(index=False, sep='\t')

# st.download_button(
#     label="Download CSV",
#     data=csv,
#     file_name="cleaned_case_data.csv",
#     mime="text/csv"
# )

# st.download_button(
#     label="Download Tab-Separated TXT",
#     data=txt,
#     file_name="cleaned_case_data.txt",
#     mime="text/plain"
# )


# st.download_button(
#     label="Download Tab-Separated TXT",
#     data=txt,
#     file_name="cleaned_case_data.txt",
#     mime="text/plain"
# )


















# import streamlit as st
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns

# # Filepath
# filepath = "CASES.xlsx"

# # Create the DataFrame with error handling
# try:
#    df = pd.read_excel(filepath, engine='openpyxl')  # Use 'openpyxl' for reading Excel files
# except Exception as e:
#     st.error(f"Error loading Excel file: {e}")
#     st.stop()  # Stop further execution if the file can't be loaded

# # Ensure necessary columns exist
# required_columns = ['Date Created', 'Date Modified', 'Assigned to', 'Case Number', 'Status', 'Online Users', 'General complaint', 'Agent']
# missing_columns = [col for col in required_columns if col not in df.columns]
# if missing_columns:
#     st.error(f"Missing columns: {', '.join(missing_columns)}")
#     st.stop()

# # Convert 'Date Created' and 'Date Modified' to datetime with error handling
# df['Date Created'] = pd.to_datetime(df['Date Created'], errors='coerce')  # Coerce errors if any
# df['Date Modified'] = pd.to_datetime(df['Date Modified'], errors='coerce')

# # Drop rows with NaT (invalid datetime)
# df = df.dropna(subset=['Date Created', 'Date Modified'])

# # Calculate time difference between 'Date Created' and 'Date Modified'
# df['Time Taken'] = df['Date Modified'] - df['Date Created']

# # Human-readable time difference
# df['Time Taken (Readable)'] = df['Time Taken'].apply(lambda x: f"{x.days} days, {x.seconds // 3600} hours, {(x.seconds // 60) % 60} minutes")

# # Merge issue-related columns
# df['Merged'] = df['Online Users'].fillna('') + ' ' + df['General complaint'].fillna('') + ' ' + df['Agent'].fillna('')

# # Drop unnecessary columns
# df_cleaned = df.drop(columns=['Online Users', 'General complaint', 'Agent'])

# # Streamlit UI
# st.title("Case Management Dashboard")
# st.write("This dashboard displays the case management data and insights.")

# # Sidebar search: Search by 'Assigned to'
# assigned_to_search = st.sidebar.selectbox('Search by Assigned to:', df['Assigned to'].unique())

# # Filter dataset based on selected 'Assigned to'
# filtered_df = df_cleaned[df_cleaned['Assigned to'] == assigned_to_search]

# # Display filtered results for all cases assigned to the selected person
# st.sidebar.subheader(f"All Cases for {assigned_to_search}")
# st.sidebar.write(filtered_df[['Case Number', 'Status', 'Time Taken (Readable)']])

# # Sidebar search: Filter by Status (Assigned or Escalated)
# status_filter = st.sidebar.selectbox('Filter by Status:', ['All', 'Assigned', 'Escalated'])
# if status_filter != 'All':
#     filtered_status_df = filtered_df[filtered_df['Status'] == status_filter]
# else:
#     filtered_status_df = filtered_df

# st.sidebar.subheader(f"Filtered Cases: {status_filter}")
# st.sidebar.dataframe(filtered_status_df)

# # Bar chart of Case Status Distribution
# st.subheader("Case Status Distribution")
# status_counts = df['Status'].value_counts()
# fig, ax = plt.subplots()
# sns.barplot(x=status_counts.index, y=status_counts.values, ax=ax, palette="viridis")
# ax.set_title('Status of Cases')
# ax.set_ylabel('Number of Cases')
# st.pyplot(fig)

# # Remove rows with blank or NaN 'Merged' values for issue count
# df_cleaned_non_na = df_cleaned[df_cleaned['Merged'].notna() & (df_cleaned['Merged'] != '')]

# # Count the occurrence of each unique issue (from 'Merged' column)
# issue_counts = df_cleaned_non_na['Merged'].value_counts()

# # Line chart of issue count over time (based on 'Date Created')
# st.subheader("Issue Count Over Time")
# issue_count_by_date = df_cleaned_non_na.groupby(df['Date Created'].dt.date)['Merged'].count()

# fig2, ax2 = plt.subplots()
# issue_count_by_date.plot(kind='line', ax=ax2, color='blue', marker='o')
# ax2.set_title('Issue Count Over Time')
# ax2.set_xlabel('Date')
# ax2.set_ylabel('Issue Count')
# st.pyplot(fig2)

# # Display the total count of each issue
# st.subheader("Total Count of Each Issue")
# st.write(issue_counts)

# # Option to download the cleaned dataset
# st.subheader("Download Cleaned Data")
# csv = df_cleaned.to_csv(index=False)
# txt = df_cleaned.to_csv(index=False, sep='\t')

# st.download_button(
#     label="Download CSV",
#     data=csv,
#     file_name="cleaned_case_data.csv",
#     mime="text/csv"
# )

# st.download_button(
#     label="Download Tab-Separated TXT",
#     data=txt,
#     file_name="cleaned_case_data.txt",
#     mime="text/plain"
# )
















# import streamlit as st
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns

# # Filepath
# filepath = "CASES.xlsx"

# # Create the DataFrame with error handling
# try:
#    df = pd.read_excel(filepath, engine='openpyxl')  # Use 'openpyxl' for reading Excel files

# except Exception as e:
#     st.error(f"Error loading Excel file: {e}")
#     st.stop()  # Stop further execution if the file can't be loaded

# # Display the columns in the Excel file
# # st.write("Columns in the Excel file:", df.columns)

# # Ensure necessary columns exist
# required_columns = ['Date Created', 'Date Modified', 'Assigned to', 'Case Number', 'Status', 'Online Users', 'General complaint', 'Agent']
# missing_columns = [col for col in required_columns if col not in df.columns]
# if missing_columns:
#     st.error(f"Missing columns: {', '.join(missing_columns)}")
#     st.stop()

# # Convert 'Date Created' and 'Date Modified' to datetime with error handling
# df['Date Created'] = pd.to_datetime(df['Date Created'], errors='coerce')  # Coerce errors if any
# df['Date Modified'] = pd.to_datetime(df['Date Modified'], errors='coerce')

# # Drop rows with NaT (invalid datetime)
# df = df.dropna(subset=['Date Created', 'Date Modified'])

# # Calculate time difference between 'Date Created' and 'Date Modified'
# df['Time Taken'] = df['Date Modified'] - df['Date Created']

# # Human-readable time difference
# df['Time Taken (Readable)'] = df['Time Taken'].apply(lambda x: f"{x.days} days, {x.seconds // 3600} hours, {(x.seconds // 60) % 60} minutes")

# # Merge issue-related columns
# df['Merged'] = df['Online Users'].fillna('') + ' ' + df['General complaint'].fillna('') + ' ' + df['Agent'].fillna('')

# # Drop unnecessary columns
# df_cleaned = df.drop(columns=['Online Users', 'General complaint', 'Agent'])

# # Streamlit UI
# st.title("Case Management Dashboard")
# st.write("This dashboard displays the case management data and insights.")

# # Sidebar search: Search by 'Assigned to'
# assigned_to_search = st.sidebar.selectbox('Search by Assigned to:', df['Assigned to'].unique())

# # Filter dataset based on selected 'Assigned to'
# filtered_df = df_cleaned[df_cleaned['Assigned to'] == assigned_to_search]

# # Display filtered results
# st.sidebar.subheader(f"Status for {assigned_to_search}")
# st.sidebar.write(filtered_df[['Case Number', 'Status', 'Time Taken (Readable)']])

# # Filter Data based on Status
# status_filter = st.sidebar.selectbox('Filter by Status:', df['Status'].unique())
# filtered_status_df = df_cleaned[df_cleaned['Status'] == status_filter]
# st.sidebar.subheader(f"Filtered Cases: {status_filter}")
# st.sidebar.dataframe(filtered_status_df)

# # Bar chart of Case Status Distribution
# st.subheader("Case Status Distribution")
# status_counts = df['Status'].value_counts()
# fig, ax = plt.subplots()
# sns.barplot(x=status_counts.index, y=status_counts.values, ax=ax, palette="viridis")
# ax.set_title('Status of Cases')
# ax.set_ylabel('Number of Cases')
# st.pyplot(fig)

# # Remove rows with blank or NaN 'Merged' values for issue count
# df_cleaned_non_na = df_cleaned[df_cleaned['Merged'].notna() & (df_cleaned['Merged'] != '')]

# # Count the occurrence of each unique issue (from 'Merged' column)
# issue_counts = df_cleaned_non_na['Merged'].value_counts()

# # Line chart of issue count over time (based on 'Date Created')
# st.subheader("Issue Count Over Time")
# issue_count_by_date = df_cleaned_non_na.groupby(df['Date Created'].dt.date)['Merged'].count()

# fig2, ax2 = plt.subplots()
# issue_count_by_date.plot(kind='line', ax=ax2, color='blue', marker='o')
# ax2.set_title('Issue Count Over Time')
# ax2.set_xlabel('Date')
# ax2.set_ylabel('Issue Count')
# st.pyplot(fig2)

# # Display the total count of each issue
# st.subheader("Total Count of Each Issue")
# st.write(issue_counts)

# # Option to download the cleaned dataset
# st.subheader("Download Cleaned Data")
# csv = df_cleaned.to_csv(index=False)
# txt = df_cleaned.to_csv(index=False, sep='\t')

# st.download_button(
#     label="Download CSV",
#     data=csv,
#     file_name="cleaned_case_data.csv",
#     mime="text/csv"
# )

# st.download_button(
#     label="Download Tab-Separated TXT",
#     data=txt,
#     file_name="cleaned_case_data.txt",
#     mime="text/plain"
# )







# import streamlit as st
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns

# # Filepath
# filepath = "CASES.xlsx"

# # Create the DataFrame with error handling
# try:
#    df = pd.read_excel(filepath, engine='openpyxl')  # Use 'openpyxl' for reading Excel files
# except Exception as e:
#     st.error(f"Error loading Excel file: {e}")
#     st.stop()  # Stop further execution if the file can't be loaded

# # Ensure necessary columns exist
# required_columns = ['Date Created', 'Date Modified', 'Assigned to', 'Case Number', 'Status', 'Online Users', 'General complaint', 'Agent']
# missing_columns = [col for col in required_columns if col not in df.columns]
# if missing_columns:
#     st.error(f"Missing columns: {', '.join(missing_columns)}")
#     st.stop()

# # Convert 'Date Created' and 'Date Modified' to datetime with error handling
# df['Date Created'] = pd.to_datetime(df['Date Created'], errors='coerce')  # Coerce errors if any
# df['Date Modified'] = pd.to_datetime(df['Date Modified'], errors='coerce')

# # Drop rows with NaT (invalid datetime)
# df = df.dropna(subset=['Date Created', 'Date Modified'])

# # Calculate time difference between 'Date Created' and 'Date Modified'
# df['Time Taken'] = df['Date Modified'] - df['Date Created']

# # Human-readable time difference
# df['Time Taken (Readable)'] = df['Time Taken'].apply(lambda x: f"{x.days} days, {x.seconds // 3600} hours, {(x.seconds // 60) % 60} minutes")

# # Merge issue-related columns
# df['Merged'] = df['Online Users'].fillna('') + ' ' + df['General complaint'].fillna('') + ' ' + df['Agent'].fillna('')

# # Drop unnecessary columns
# df_cleaned = df.drop(columns=['Online Users', 'General complaint', 'Agent'])

# # Streamlit UI
# st.title("Case Management Dashboard")
# st.write("This dashboard displays the case management data and insights.")

# # Sidebar search: Search by 'Assigned to'
# assigned_to_search = st.sidebar.selectbox('Search by Assigned to:', df['Assigned to'].unique())

# # Filter dataset based on selected 'Assigned to'
# filtered_df = df_cleaned[df_cleaned['Assigned to'] == assigned_to_search]

# # Display filtered results for all cases assigned to the selected person
# st.sidebar.subheader(f"All Cases for {assigned_to_search}")
# st.sidebar.write(filtered_df[['Case Number', 'Status', 'Time Taken (Readable)']])

# # Sidebar search: Filter by 'Status' (Open or Resolved)
# status_filter = st.sidebar.selectbox('Filter by Status:', ['All', 'Resolved', 'Open'])
# if status_filter != 'All':
#     filtered_status_df = filtered_df[filtered_df['Status'] == status_filter]
# else:
#     filtered_status_df = filtered_df

# st.sidebar.subheader(f"Filtered Cases: {status_filter}")
# st.sidebar.dataframe(filtered_status_df)

# # Bar chart of Case Status Distribution
# st.subheader("Case Status Distribution")
# status_counts = df['Status'].value_counts()
# fig, ax = plt.subplots()
# sns.barplot(x=status_counts.index, y=status_counts.values, ax=ax, palette="viridis")
# ax.set_title('Status of Cases')
# ax.set_ylabel('Number of Cases')
# st.pyplot(fig)

# # Remove rows with blank or NaN 'Merged' values for issue count
# df_cleaned_non_na = df_cleaned[df_cleaned['Merged'].notna() & (df_cleaned['Merged'] != '')]

# # Count the occurrence of each unique issue (from 'Merged' column)
# issue_counts = df_cleaned_non_na['Merged'].value_counts()

# # Line chart of issue count over time (based on 'Date Created')
# st.subheader("Issue Count Over Time")
# issue_count_by_date = df_cleaned_non_na.groupby(df['Date Created'].dt.date)['Merged'].count()

# fig2, ax2 = plt.subplots()
# issue_count_by_date.plot(kind='line', ax=ax2, color='blue', marker='o')
# ax2.set_title('Issue Count Over Time')
# ax2.set_xlabel('Date')
# ax2.set_ylabel('Issue Count')
# st.pyplot(fig2)

# # Display the total count of each issue
# st.subheader("Total Count of Each Issue")
# st.write(issue_counts)

# # Option to download the cleaned dataset
# st.subheader("Download Cleaned Data")
# csv = df_cleaned.to_csv(index=False)
# txt = df_cleaned.to_csv(index=False, sep='\t')

# st.download_button(
#     label="Download CSV",
#     data=csv,
#     file_name="cleaned_case_data.csv",
#     mime="text/csv"
# )

# st.download_button(
#     label="Download Tab-Separated TXT",
#     data=txt,
#     file_name="cleaned_case_data.txt",
#     mime="text/plain"
# )

















# import streamlit as st
# import pandas as pd

# # Define the data (same as before)
# data = {
#     "Assigned to": ["Sarah Eguaomon", "Adeyemi Akanji", "Abdul-Qudus Adekoya", "Adewale Bajela", "Adewale Bajela", "Adewale Bajela", "Adewale Bajela", "Adewale Bajela", "Adewale Bajela"],
#     "Case Number": ["CRM244032", "CRM244122", "CRM243794", "CRM244094", "CRM244116", "CRM244058", "CRM243891", "CRM243835", "CRM243810"],
#     "Customer Id": ["18672525", "21085295", "", "21848666", "22091482", "22160887", "6223131", "17111523", "7323860"],
#     "Date Created": ["01/19/2025 04:56pm", "01/19/2025 08:38pm", "01/19/2025 10:41am", "01/19/2025 07:08pm", "01/19/2025 08:22pm", "01/19/2025 05:40pm", "01/19/2025 12:59pm", "01/19/2025 11:49am", "01/19/2025 11:08am"],
#     "Date Modified": ["01/22/2025 06:19pm", "01/22/2025 04:57pm", "01/22/2025 01:50pm", "01/22/2025 11:55am", "01/22/2025 11:54am", "01/22/2025 11:54am", "01/22/2025 11:54am", "01/22/2025 11:53am", "01/22/2025 11:53am"],
#     "Lead Consumer": ["Mr. ABDULLAHI BILAL", "API Leads", "API Leads", "Mr. OTOIKHILA BLESSINGS", "Mr. STEPHEN EFENA", "Dauda Funlonsho", "Mr. CHRISTOPHER CHRISTOPHER AFOR KINGSLEY", "KELVIN AGU", "Mr. ABDULWAHAB AMOTO HASSAN"],
#     "Online Users": ["Self-Exclusion/ Self Disabled account", "", "", "", "", "Change of any account details", "Sport Bonus Complaint", "", "Mistakenly crediting another user, a deposit"],
#     "General complaint": ["", "Betslip Complaints", "", "General inquiries", "Betslip Complaints", "", "", "General inquiries", ""],
#     "Agent": ["", "", "", "", "", "", "", "", ""],
#     "Status": ["Resolved", "In progress", "Opened", "Resolved", "Resolved", "Resolved", "Resolved", "Resolved", "Resolved"]
# }

# # Create the DataFrame
# df = pd.DataFrame(data)

# # Convert 'Date Created' and 'Date Modified' to datetime
# df['Date Created'] = pd.to_datetime(df['Date Created'], format="%m/%d/%Y %I:%M%p")
# df['Date Modified'] = pd.to_datetime(df['Date Modified'], format="%m/%d/%Y %I:%M%p")

# # Calculate the time difference between 'Date Created' and 'Date Modified'
# df['Time Taken'] = df['Date Modified'] - df['Date Created']

# # Create a human-readable format for the time difference (e.g., hours and minutes)
# df['Time Taken (Readable)'] = df['Time Taken'].apply(lambda x: f"{x.days} days, {x.seconds // 3600} hours, {(x.seconds // 60) % 60} minutes")

# # Merge the 'Online Users', 'General complaint', and 'Agent' columns into a single column
# df['Merged'] = df['Online Users'].fillna('') + ' ' + df['General complaint'].fillna('') + ' ' + df['Agent'].fillna('')

# # Drop the original columns if no longer needed
# df_cleaned = df.drop(columns=['Online Users', 'General complaint', 'Agent'])

# # Streamlit UI
# st.title("Case Management Dashboard")
# st.write("This dashboard displays the case management data and insights.")

# # Display the dataset in a table
# st.subheader("Case Data with Time Taken")
# st.dataframe(df_cleaned)

# # Add Search Bar using Selectbox to filter by Assigned to
# assigned_to_search = st.selectbox('Search by Assigned to:', df['Assigned to'].unique())

# # Filter dataset based on the selected Assigned to value
# filtered_df = df_cleaned[df_cleaned['Assigned to'] == assigned_to_search]

# # Display filtered results
# st.subheader(f"Status for {assigned_to_search}")
# st.write(filtered_df[['Case Number', 'Status', 'Time Taken (Readable)']])

# # Option to filter by Status (resolved, open, etc.)
# status_filter = st.selectbox('Filter by Status:', df['Status'].unique())
# filtered_status_df = df_cleaned[df_cleaned['Status'] == status_filter]
# st.subheader(f"Filtered Cases: {status_filter}")
# st.dataframe(filtered_status_df)

# # Remove rows with blank or NaN 'Merged' values
# df_cleaned_non_na = df_cleaned[df_cleaned['Merged'].notna() & (df_cleaned['Merged'] != '')]

# # Count the occurrence of each unique issue (from 'Merged' column)
# issue_counts = df_cleaned_non_na['Merged'].value_counts()

# # Display the total count of each issue type
# st.subheader("Total Count of Each Issue")
# st.write(issue_counts)

# # Option to download the cleaned dataset
# st.subheader("Download Cleaned Data")
# csv = df_cleaned.to_csv(index=False)
# txt = df_cleaned.to_csv(index=False, sep='\t')

# st.download_button(
#     label="Download CSV",
#     data=csv,
#     file_name="cleaned_case_data.csv",
#     mime="text/csv"
# )

# st.download_button(
#     label="Download TXT",
#     data=txt,
#     file_name="cleaned_case_data.txt",
#     mime="text/plain"
# )



















# import streamlit as st
# import pandas as pd
# import matplotlib.pyplot as plt

# # Sample leave dataset
# leave_data = {
#     "Employee Name": ["John Smith", "Alice Johnson", "Robert Brown", "Emily Davis", "Michael Wilson"],
#     "Leave Type": ["Annual", "Sick", "Casual", "Maternity", "Annual"],
#     "Days Taken": [10, 5, 2, 90, 15],
#     "Remaining Days": [5, 10, 8, 0, 5],
# }

# # Convert data to DataFrame
# df = pd.DataFrame(leave_data)

# # Calculate total leave taken and remaining
# total_days_taken = df["Days Taken"].sum()
# total_remaining_days = df["Remaining Days"].sum()

# # Group by Leave Type to show aggregate leave data
# leave_summary = df.groupby("Leave Type")[["Days Taken", "Remaining Days"]].sum().reset_index()

# # Streamlit app
# st.title("Leave Management Dashboard")
# st.subheader("Leave Details for Employees")

# # Display the main DataFrame
# st.dataframe(df)

# # Display totals
# st.markdown("### Totals")
# st.write(f"**Total Days Taken:** {total_days_taken}")
# st.write(f"**Total Remaining Days:** {total_remaining_days}")

# # Display the summary table
# st.markdown("### Leave Summary by Type")
# st.dataframe(leave_summary)

# # Plotting a bar chart for leave type distribution
# st.markdown("### Leave Type Distribution")

# # Create a figure
# fig, ax = plt.subplots()
# leave_summary.set_index('Leave Type')[['Days Taken']].plot(kind='bar', legend=True, ax=ax)
# ax.set_title('Leave Type Distribution (Days Taken)')
# ax.set_ylabel('Days Taken')

# # Pass the figure to st.pyplot()
# st.pyplot(fig)

# # Add some interactivity: Filter by leave type
# st.markdown("### Filter Leave Data by Type")
# leave_type = st.selectbox("Select Leave Type:", df["Leave Type"].unique())
# filtered_data = df[df["Leave Type"] == leave_type]
# st.write(f"Employees with **{leave_type} Leave**:")
# st.dataframe(filtered_data)








# import streamlit as st
# import pandas as pd

# # Sample leave dataset
# leave_data = {
#     "Employee Name": ["John Smith", "Alice Johnson", "Robert Brown", "Emily Davis", "Michael Wilson"],
#     "Leave Type": ["Annual", "Sick", "Casual", "Maternity", "Annual"],
#     "Days Taken": [10, 5, 2, 90, 15],
#     "Remaining Days": [5, 10, 8, 0, 5],
# }

# # Convert data to DataFrame
# df = pd.DataFrame(leave_data)

# # Calculate total leave taken and remaining
# total_days_taken = df["Days Taken"].sum()
# total_remaining_days = df["Remaining Days"].sum()

# # Group by Leave Type to show aggregate leave data
# leave_summary = df.groupby("Leave Type")[["Days Taken", "Remaining Days"]].sum().reset_index()

# # Streamlit app
# st.title("Leave Management Dashboard")
# st.subheader("Leave Details for Employees")

# # Display the main DataFrame
# st.dataframe(df)

# # Display totals
# st.markdown("### Totals")
# st.write(f"**Total Days Taken:** {total_days_taken}")
# st.write(f"**Total Remaining Days:** {total_remaining_days}")

# # Display the summary table
# st.markdown("### Leave Summary by Type")
# st.dataframe(leave_summary)

# # Add some interactivity: Filter by leave type
# st.markdown("### Filter Leave Data by Type")
# leave_type = st.selectbox("Select Leave Type:", df["Leave Type"].unique())
# filtered_data = df[df["Leave Type"] == leave_type]
# st.write(f"Employees with **{leave_type} Leave**:")
# st.dataframe(filtered_data)





















# import streamlit as st
# import pandas as pd

# # Employee leave dataset
# data = {
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
#     'Reason': ['Annual Leave', 'Personal Leave', 'Sick Leave', 'Casual Leave',
#                'Annual Leave', 'Maternity Leave', 'Paternity Leave', 'Compassionate Leave', 'Study Leave']
# }
# df = pd.DataFrame(data)

# # Convert dates to datetime
# df['Start_Date'] = pd.to_datetime(df['Start_Date'])
# df['End_Date'] = pd.to_datetime(df['End_Date'])
# df['Month'] = df['Start_Date'].dt.month  # Extract month as a number

# # Map month numbers to names
# month_name_map = {
#     1: 'January', 2: 'February', 3: 'March', 4: 'April',
#     5: 'May', 6: 'June', 7: 'July', 8: 'August', 9: 'September',
#     10: 'October', 11: 'November', 12: 'December'
# }
# df['Month_Name'] = df['Month'].map(month_name_map)

# # Group data by month and calculate total leave taken
# monthly_trends = df.groupby('Month_Name', sort=False)['Leave_Taken'].sum().reset_index()

# # Sort by month order for correct display
# month_order = list(month_name_map.values())  # Ensure order by month names
# monthly_trends['Month_Name'] = pd.Categorical(monthly_trends['Month_Name'], categories=month_order, ordered=True)
# monthly_trends = monthly_trends.sort_values('Month_Name')

# # Determine the month with the most and least leave taken
# most_leave_month = monthly_trends.loc[monthly_trends['Leave_Taken'].idxmax()]
# least_leave_month = monthly_trends.loc[monthly_trends['Leave_Taken'].idxmin()]

# # Streamlit interface
# st.title("Employee Leave Dashboard")

# # Display monthly trends in a table format
# st.write("### Monthly Leave Trends")
# st.table(monthly_trends.rename(columns={'Month_Name': 'Month', 'Leave_Taken': 'Total Leave Taken'}))

# # Highlight the months with most and least leave taken
# st.write(f"The month with the **most leave taken** is **{most_leave_month['Month_Name']}** with **{most_leave_month['Leave_Taken']} days**.")
# st.write(f"The month with the **least leave taken** is **{least_leave_month['Month_Name']}** with **{least_leave_month['Leave_Taken']} days**.")

# # Visualize the monthly trends
# st.write("### Leave Trend Visualization")
# st.line_chart(monthly_trends.set_index('Month_Name')['Leave_Taken'])











# import streamlit as st
# import pandas as pd

# # Employee leave dataset
# data = {
#     'Employee_ID': ['E101', 'E102', 'E103', 'E104', 'E105', 'E106', 'E107', 'E108', 'E109'],
#     'Employee_Name': ['John Doe', 'Jane Smith', 'Mike Johnson', 'Emily Davis', 'James Wilson',
#                       'Sarah Brown', 'Robert Green', 'Linda White', 'David Black'],
#     'Start_Date': ['2024-01-05', '2024-01-01', '2024-03-15', '2024-04-05', '2024-05-10',
#                    '2024-06-01', '2024-07-01', '2024-08-10', '2024-09-15'],
#     'End_Date': ['2024-01-10', '2024-02-03', '2024-03-20', '2024-04-07', '2024-05-14',
#                  '2024-06-30', '2024-07-05', '2024-08-20', '2024-09-25'],
#     'Leave_Taken': [5, 2, 5, 2, 4, 30, 5, 10, 10],
#     'Total_Annual_Leave': [20, 20, 20, 20, 20, 20, 20, 20, 20],
#     'Leave_Remaining': [15, 18, 15, 18, 16, -10, 15, 10, 10],
#     'Reason': ['Annual Leave', 'Personal Leave', 'Sick Leave', 'Casual Leave',
#                'Annual Leave', 'Maternity Leave', 'Paternity Leave', 'Compassionate Leave', 'Study Leave']
# }
# df = pd.DataFrame(data)


# # Convert dates to datetime
# df['Start_Date'] = pd.to_datetime(df['Start_Date'])
# df['End_Date'] = pd.to_datetime(df['End_Date'])
# df['Month'] = df['Start_Date'].dt.month  # Extract month as a number

# # Map month numbers to names
# month_name_map = {
#     1: 'January', 2: 'February', 3: 'March', 4: 'April',
#     5: 'May', 6: 'June', 7: 'July', 8: 'August', 9: 'September',
#     10: 'October', 11: 'November', 12: 'December'
# }
# df['Month_Name'] = df['Month'].map(month_name_map)

# # Group data by month and calculate total leave taken
# monthly_trends = df.groupby('Month_Name', sort=False)['Leave_Taken'].sum().reset_index()

# # Sort by month order for correct display
# month_order = list(month_name_map.values())  # Ensure order by month names
# monthly_trends['Month_Name'] = pd.Categorical(monthly_trends['Month_Name'], categories=month_order, ordered=True)
# monthly_trends = monthly_trends.sort_values('Month_Name')

# # Determine the month with the most and least leave taken
# most_leave_month = monthly_trends.loc[monthly_trends['Leave_Taken'].idxmax()]
# least_leave_month = monthly_trends.loc[monthly_trends['Leave_Taken'].idxmin()]

# # Streamlit interface
# st.title("Employee Leave Dashboard")

# st.write(df)

# # Display monthly trends in a table format
# st.write("### Monthly Leave Trends")
# st.table(monthly_trends.rename(columns={'Month_Name': 'Month', 'Leave_Taken': 'Total Leave Taken'}))

# # Highlight the months with most and least leave taken
# st.write(f"The month with the **most leave taken** is **{most_leave_month['Month_Name']}** with **{most_leave_month['Leave_Taken']} days**.")
# st.write(f"The month with the **least leave taken** is **{least_leave_month['Month_Name']}** with **{least_leave_month['Leave_Taken']} days**.")

# # Visualize the monthly trends
# st.line_chart(monthly_trends.set_index('Month_Name')['Leave_Taken'])










# import streamlit as st
# import pandas as pd

# # Employee leave dataset
# data = {
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
#     'Reason': ['Annual Leave', 'Personal Leave', 'Sick Leave', 'Casual Leave', 
#                'Annual Leave', 'Maternity Leave', 'Paternity Leave', 'Compassionate Leave', 'Study Leave']
# }
# df = pd.DataFrame(data)

# # Convert dates to datetime
# df['Start_Date'] = pd.to_datetime(df['Start_Date'])
# df['End_Date'] = pd.to_datetime(df['End_Date'])
# df['Month'] = df['Start_Date'].dt.strftime('%Y-%m')  # Extracting month in 'YYYY-MM' format

# # Multiselect widget for filtering leave types
# leave_types = df['Reason'].unique().tolist()
# selected_leave_types = st.multiselect('Select Leave Types to Display:', leave_types, default=leave_types)

# # Filter dataset based on selected leave types
# filtered_df = df[df['Reason'].isin(selected_leave_types)]

# # Display the filtered dataset
# st.write(f"Filtered Leave Data ({len(filtered_df)} records):")
# st.dataframe(filtered_df)

# # Metrics: Total leave taken and average leave remaining
# st.write(f"Total Leave Taken: {filtered_df['Leave_Taken'].sum()}")
# st.write(f"Average Leave Remaining: {filtered_df['Leave_Remaining'].mean():.2f}")

# # Employee statistics
# most_leave = df.loc[df['Leave_Taken'].idxmax()]
# least_leave = df.loc[df['Leave_Taken'].idxmin()]
# most_remaining = df.loc[df['Leave_Remaining'].idxmax()]
# least_remaining = df.loc[df['Leave_Remaining'].idxmin()]

# st.write("### Employee Statistics")
# st.write(f"Employee with the Most Leave Taken: {most_leave['Employee_Name']} ({most_leave['Leave_Taken']} days)")
# st.write(f"Employee with the Least Leave Taken: {least_leave['Employee_Name']} ({least_leave['Leave_Taken']} days)")
# st.write(f"Employee with the Most Leave Remaining: {most_remaining['Employee_Name']} ({most_remaining['Leave_Remaining']} days)")
# st.write(f"Employee with the Least Leave Remaining: {least_remaining['Employee_Name']} ({least_remaining['Leave_Remaining']} days)")

# # Count of employees by leave type
# leave_type_counts = filtered_df['Reason'].value_counts()
# st.write("### Count of Employees Taking Each Type of Leave")
# st.bar_chart(leave_type_counts)

# # Employee selection based on leave type
# selected_reason = st.selectbox("Select a Leave Type to View Employees:", leave_types)
# employees_on_leave = df[df['Reason'] == selected_reason][['Employee_ID', 'Employee_Name', 'Leave_Taken']]
# st.write(f"Employees on {selected_reason}:")
# st.dataframe(employees_on_leave)

# # Monthly trends for leaves taken
# monthly_trends = df.groupby('Month')['Leave_Taken'].sum().reset_index()

# # Determine the month with the most and least leave taken
# most_leave_month = monthly_trends.loc[monthly_trends['Leave_Taken'].idxmax()]
# least_leave_month = monthly_trends.loc[monthly_trends['Leave_Taken'].idxmin()]

# # Display monthly trends
# st.write("### Monthly Trends for Leaves Taken")
# st.line_chart(monthly_trends.set_index('Month'))

# # Display months with most and least leave taken
# st.write(f"The month with the most leave taken is **{most_leave_month['Month']}** with **{most_leave_month['Leave_Taken']} days**.")
# st.write(f"The month with the least leave taken is **{least_leave_month['Month']}** with **{least_leave_month['Leave_Taken']} days**.")













# import streamlit as st
# import pandas as pd

# # Sample leave data with Leave Type
# leave_data = {
#     "Employee": ["John Doe", "Jane Smith", "Alex Taylor", "Emily Brown"],
#     "Start Date": ["2025-01-15", "2025-01-18", "2025-01-20", "2025-01-22"],
#     "End Date": ["2025-01-20", "2025-01-25", "2025-01-22", "2025-01-27"],
#     "Leave Type": ["Annual Leave", "Sick Leave", "Casual Leave", "Maternity Leave"],
# }

# # Convert to DataFrame
# df = pd.DataFrame(leave_data)

# # Convert dates to datetime
# df["Start Date"] = pd.to_datetime(df["Start Date"])
# df["End Date"] = pd.to_datetime(df["End Date"])

# # Function to get weekdays within a date range
# def get_weekdays(start_date, end_date):
#     if start_date > end_date:
#         return []  # Return empty list for invalid date ranges
#     all_dates = pd.date_range(start=start_date, end=end_date)
#     weekdays = all_dates[all_dates.weekday < 5]  # Exclude weekends
#     return weekdays

# # Apply the function to calculate weekdays and leave taken
# df["Weekdays"] = df.apply(lambda row: get_weekdays(row["Start Date"], row["End Date"]), axis=1)
# df["Leave Taken"] = df["Weekdays"].apply(len)  # Count weekdays

# # Drop the 'Weekdays' column to simplify display
# df_display = df[["Employee", "Start Date", "End Date", "Leave Type", "Leave Taken"]]

# # Display the enhanced DataFrame
# st.title("Leave Tracker")
# st.write("Below is the leave data with the calculated number of weekdays:")
# st.dataframe(df_display)






















# import streamlit as st
# import pandas as pd

# # Sample leave dataset
# leave_data = {
#     "Employee Name": ["John Smith", "Alice Johnson", "Robert Brown", "Emily Davis", "Michael Wilson"],
#     "Leave Type": ["Annual", "Sick", "Casual", "Maternity", "Annual"],
#     "Days Taken": [10, 5, 2, 90, 15],
#     "Remaining Days": [5, 10, 8, 0, 5]
# }
# df = pd.DataFrame(leave_data)

# # Display the image with a fixed width for proper sizing
# st.image("Bet9ja_logo.jpeg", width=200, caption="Our Logo")

# # Title for the application
# st.title("Employee Leave Management System")

# # Sidebar for input with form
# with st.sidebar:
#     st.header("Search Here")
#     with st.form("search_form"):
#         employee = st.text_input("Enter your name:")
#         search_button = st.form_submit_button("Click to Search")

# # Main page content
# if search_button:  # This is triggered by the form submit (button click or Enter key)
#     if employee:
#         df2 = df[df['Employee Name'].str.contains(employee, case=False, na=False)]
#         if not df2.empty:
#             st.subheader("Search Results")
#             st.write(df2)
#         else:
#             st.warning("No record found.")
#     else:
#         st.warning("Please enter a name.")










# import streamlit as st
# import pandas as pd

# # Sample leave dataset
# leave_data = {
#     "Employee Name": ["John Smith", "Alice Johnson", "Robert Brown", "Emily Davis", "Michael Wilson"],
#     "Leave Type": ["Annual", "Sick", "Casual", "Maternity", "Annual"],
#     "Days Taken": [10, 5, 2, 90, 15],
#     "Remaining Days": [5, 10, 8, 0, 5]
# }
# df = pd.DataFrame(leave_data)

# # Display the image with a fixed width for proper sizing
# st.image("Bet9ja_logo.jpeg", width=200, caption="Our Logo")

# # Title for the application
# st.title("Employee Leave Management System")

# # Sidebar for input with form
# with st.sidebar:
#     st.header("Search Here")
    
#     # Create a unique key for the input field
#     reset_key = st.session_state.get("reset_key", 0)  # Get the reset key from session state
    
#     # Form for input with dynamic key
#     with st.form("search_form"):
#         employee = st.text_input("Enter your name:", key=f"employee_input_{reset_key}")
#         search_button = st.form_submit_button("Click to Search")
#         reset_button = st.form_submit_button("Reset")  # Reset button

#         # Reset logic
#         if reset_button:
#             # Increment the reset key to make the key unique and reset the input field
#             st.session_state["reset_key"] = reset_key + 1  # Update session state with a new reset key

# # Main page content
# if search_button:  # Triggered by the form submit (button click or Enter key)
#     if employee:
#         df2 = df[df['Employee Name'].str.contains(employee, case=False, na=False)]
#         if not df2.empty:
#             st.subheader("Search Results")
#             st.write(df2)
#         else:
#             st.warning("No record found.")
#     else:
#         st.warning("Please enter a name.")

















# import streamlit as st
# import pandas as pd

# # Sample leave dataset
# leave_data = {
#     "Employee Name": ["John Smith", "Alice Johnson", "Robert Brown", "Emily Davis", "Michael Wilson"],
#     "Leave Type": ["Annual", "Sick", "Casual", "Maternity", "Annual"],
#     "Days Taken": [10, 5, 2, 90, 15],
#     "Remaining Days": [5, 10, 8, 0, 5]
# }
# df = pd.DataFrame(leave_data)

# # Display the image with a fixed width for proper sizing
# st.image("Bet9ja_logo.jpeg", width=200, caption="Our Logo")

# # Title for the application
# st.title("Employee Leave Management System")

# with st.sidebar:
#     st.header("search here")

#     employee = st.text_input("enter your name:")

# if st.button("click to search"):
#     if employee:
#         df2= df[df['Employee Name'].str.contains(employee, case=False, na=False)]

#         if not df2.empty:

#             st.subheader("see search result below")
#             st.write(df2)
#         else:
#             st.warning("no record founds")
#     else:
#         st.write("PLease enter the correct name")            



























# import pandas as pd
# import streamlit as st

# # Sample DataFrame
# data = {
#     'Name': ['Alice', 'Bob', 'Charlie', 'Diana', 'Edward', 'Fiona', 'George'],
#     'Leave_Taken': [10, 15, 7, 12, 20, 5, 18],
# }

# leave_df = pd.DataFrame(data)


# st.title("My first project")
# st.dataframe(leave_df.head(4))


# top4 = leave_df.nlargest(4, 'Leave_Taken')

# least4 = leave_df.nsmallest(4, 'Leave_Taken')

# col1, col2 = st.columns(2)

# with col1:
#     st.header("Top 4 users")
#     st.dataframe(top4)

# with col2:
#     st.header("Least 4 user")
#     st.dataframe(least4)    

# with st.sidebar:
#     st.header("Search Employee")
#     employee_name = st.text_input("Enter Employee Name:")
#     search_button = st.button("Search")

# # Main content
# if search_button:
#     if employee_name:
#         # Filter dataset
#         filtered_df = leave_df[
#             leave_df["Name"].str.contains(employee_name, case=False, na=False)
#         ]
#         if not filtered_df.empty:
#             st.subheader("Search Results")
#             st.write(filtered_df)
#         else:
#             st.warning("No results found for the given name.")
#     else:
#         st.warning("Please enter a name to search.")






# import streamlit as st
# import pandas as pd

# # Sample leave dataset
# leave_data = {
#     "Employee Name": ["John Smith", "Alice Johnson", "Robert Brown", "Emily Davis", "Michael Wilson"],
#     "Leave Type": ["Annual", "Sick", "Casual", "Maternity", "Annual"],
#     "Days Taken": [10, 5, 2, 90, 15],
#     "Remaining Days": [5, 10, 8, 0, 5]
# }
# leave_df = pd.DataFrame(leave_data)

# # Streamlit app
# st.title("Employee Leave Search")

# # Sidebar input for search query
# with st.sidebar:
#     st.header("Search Employee")
#     employee_name = st.text_input("Enter Employee Name:")
#     search_button = st.button("Search")

# # Main content
# if search_button:
#     if employee_name:
#         # Filter dataset
#         filtered_df = leave_df[
#             leave_df["Employee Name"].str.contains(employee_name, case=False, na=False)
#         ]
#         if not filtered_df.empty:
#             st.subheader("Search Results")
#             st.write(filtered_df)
#         else:
#             st.warning("No results found for the given name.")
#     else:
#         st.warning("Please enter a name to search.")




















# import streamlit as st
# import pandas as pd

# # Employee leave dataset
# data = {
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
#     'Reason': ['Annual Leave', 'Personal Leave', 'Sick Leave', 'Casual Leave', 
#                'Annual Leave', 'Maternity Leave', 'Paternity Leave', 'Compassionate Leave', 'Study Leave']
# }
# df = pd.DataFrame(data)

# # Convert date columns to datetime format
# df['Start_Date'] = pd.to_datetime(df['Start_Date'])
# df['End_Date'] = pd.to_datetime(df['End_Date'])


#  # Leave type filter
# leave_types = df['Reason'].unique().tolist()
# selected_leave_types = st.multiselect(
#     'Select Leave Types to Display:', 
#     leave_types, 
#     default=leave_types
# )

# # Date range filter
# start_date = pd.to_datetime(st.date_input('Start Date:', value=pd.to_datetime('2024-01-01')))
# end_date = pd.to_datetime(st.date_input('End Date:', value=pd.to_datetime('2024-12-31')))

# # Filter dataset
# filtered_df = df[
#     (df['Reason'].isin(selected_leave_types)) & 
#     (df['Start_Date'] >= start_date) & 
#     (df['End_Date'] <= end_date)
# ]

# # Additional metrics
# total_leave_taken = filtered_df['Leave_Taken'].sum()
# average_leave_remaining = filtered_df['Leave_Remaining'].mean()

# # Display filtered data and metrics
# st.write(f"Filtered Leave Data ({len(filtered_df)} records):")
# st.dataframe(filtered_df)

# st.write(f"**Total Leave Taken:** {total_leave_taken}")
# st.write(f"**Average Leave Remaining:** {average_leave_remaining:.2f}")

# # Option to export filtered data
# csv = filtered_df.to_csv(index=False).encode('utf-8')
# st.download_button(
#     label="Download Filtered Data as CSV",
#     data=csv,
#     file_name="filtered_leave_data.csv",
#     mime="text/csv"
# )




















# import streamlit as st
# import pandas as pd

# # Employee leave dataset
# data = {
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
#     'Reason': ['Annual Leave', 'Personal Leave', 'Sick Leave', 'Casual Leave', 
#                'Annual Leave', 'Maternity Leave', 'Paternity Leave', 'Compassionate Leave', 'Study Leave']
# }
# df = pd.DataFrame(data)

# # Convert date columns to datetime format
# df['Start_Date'] = pd.to_datetime(df['Start_Date'])
# df['End_Date'] = pd.to_datetime(df['End_Date'])

# # Expander for filtering options
# with st.expander("Filter Leave Types and View Data"):
#     # Leave type filter
#     leave_types = df['Reason'].unique().tolist()
#     selected_leave_types = st.multiselect(
#         'Select Leave Types to Display:', 
#         leave_types, 
#         default=leave_types
#     )

#     # Date range filter
#     start_date = pd.to_datetime(st.date_input('Start Date:', value=pd.to_datetime('2024-01-01')))
#     end_date = pd.to_datetime(st.date_input('End Date:', value=pd.to_datetime('2024-12-31')))

#     # Filter dataset
#     filtered_df = df[
#         (df['Reason'].isin(selected_leave_types)) & 
#         (df['Start_Date'] >= start_date) & 
#         (df['End_Date'] <= end_date)
#     ]

#     # Additional metrics
#     total_leave_taken = filtered_df['Leave_Taken'].sum()
#     average_leave_remaining = filtered_df['Leave_Remaining'].mean()

# # Display filtered data and metrics outside the expander
# st.write(f"Filtered Leave Data ({len(filtered_df)} records):")
# st.dataframe(filtered_df)

# st.write(f"**Total Leave Taken:** {total_leave_taken}")
# st.write(f"**Average Leave Remaining:** {average_leave_remaining:.2f}")

# # Option to export filtered data
# csv = filtered_df.to_csv(index=False).encode('utf-8')
# st.download_button(
#     label="Download Filtered Data as CSV",
#     data=csv,
#     file_name="filtered_leave_data.csv",
#     mime="text/csv"
# )



















# import streamlit as st
# import pandas as pd

# # Employee leave dataset
# data = {
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
#     'Reason': ['Annual Leave', 'Personal Leave', 'Sick Leave', 'Casual Leave', 
#                'Annual Leave', 'Maternity Leave', 'Paternity Leave', 'Compassionate Leave', 'Study Leave']
# }
# df = pd.DataFrame(data)

# # Multiselect widget for filtering leave types
# leave_types = df['Reason'].unique().tolist()
# selected_leave_types = st.multiselect('Select Leave Types to Display:', leave_types, default=leave_types)

# # Filter dataset based on selected leave types
# filtered_df = df[df['Reason'].isin(selected_leave_types)]

# # Calculate metrics
# leave_taken = filtered_df['Leave_Taken'].sum()
# average_leave_remaining = filtered_df['Leave_Remaining'].mean()

# # Display the filtered dataset
# st.write(f"Filtered Leave Data ({len(filtered_df)} records):")
# st.dataframe(filtered_df)

# # Separate sections for Leave_Taken and Leave_Remaining
# st.subheader("Leave Taken Metrics")
# st.write(f"Total Leave Taken: {leave_taken}")
# st.write("Detailed Leave Taken Data:")
# st.dataframe(filtered_df[['Employee_ID', 'Employee_Name', 'Leave_Taken']])

# st.subheader("Leave Remaining Metrics")
# st.write(f"Average Leave Remaining: {average_leave_remaining:.2f}")
# st.write("Detailed Leave Remaining Data:")
# st.dataframe(filtered_df[['Employee_ID', 'Employee_Name', 'Leave_Remaining']])

# if not filtered_df.empty:
#     csv= filtered_df.to_csv(index=False).encode('utf-8')
#     st.download_button(
#         label= "Download here",
#         data=csv,
#         file_name= "data.csv",
#         mime= "text/csv"
#     )


















# import streamlit as st
# import pandas as pd

# # Employee leave dataset
# data = {
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
#     'Reason': ['Annual Leave', 'Personal Leave', 'Sick Leave', 'Casual Leave', 
#                'Annual Leave', 'Maternity Leave', 'Paternity Leave', 'Compassionate Leave', 'Study Leave']
# }
# df = pd.DataFrame(data)

# # Convert dates to datetime
# df['Start_Date'] = pd.to_datetime(df['Start_Date'])
# df['End_Date'] = pd.to_datetime(df['End_Date'])
# df['Month'] = df['Start_Date'].dt.strftime('%Y-%m')  # Extracting month in 'YYYY-MM' format

# # Sidebar for filters
# st.sidebar.title("Dashboard Filters")
# leave_types = df['Reason'].unique().tolist()
# selected_leave_types = st.sidebar.multiselect('Select Leave Types:', leave_types, default=leave_types)

# # Filter dataset based on selected leave types
# filtered_df = df[df['Reason'].isin(selected_leave_types)]

# # Main Dashboard
# st.title("Employee Leave Dashboard")

# with st.expander("Filtered Leave Data"):
#     st.write(f"Filtered Leave Data ({len(filtered_df)} records):")
#     st.dataframe(filtered_df)

#     # Metrics: Total leave taken and average leave remaining
#     st.write(f"Total Leave Taken: {filtered_df['Leave_Taken'].sum()}")
#     st.write(f"Average Leave Remaining: {filtered_df['Leave_Remaining'].mean():.2f}")

# with st.expander("Employee Statistics"):
#     most_leave = df.loc[df['Leave_Taken'].idxmax()]
#     least_leave = df.loc[df['Leave_Taken'].idxmin()]
#     most_remaining = df.loc[df['Leave_Remaining'].idxmax()]
#     least_remaining = df.loc[df['Leave_Remaining'].idxmin()]

#     st.write(f"Employee with the Most Leave Taken: **{most_leave['Employee_Name']}** ({most_leave['Leave_Taken']} days)")
#     st.write(f"Employee with the Least Leave Taken: **{least_leave['Employee_Name']}** ({least_leave['Leave_Taken']} days)")
#     st.write(f"Employee with the Most Leave Remaining: **{most_remaining['Employee_Name']}** ({most_remaining['Leave_Remaining']} days)")
#     st.write(f"Employee with the Least Leave Remaining: **{least_remaining['Employee_Name']}** ({least_remaining['Leave_Remaining']} days)")

# with st.expander("Leave Type Distribution"):
#     leave_type_counts = filtered_df['Reason'].value_counts()
#     st.bar_chart(leave_type_counts)

#     selected_reason = st.selectbox("Select a Leave Type to View Employees:", leave_types)
#     employees_on_leave = df[df['Reason'] == selected_reason][['Employee_ID', 'Employee_Name', 'Leave_Taken']]
#     st.write(f"Employees on **{selected_reason}**:")
#     st.dataframe(employees_on_leave)

# with st.expander("Monthly Trends for Leaves Taken"):
#     monthly_trends = df.groupby('Month')['Leave_Taken'].sum().reset_index()

#     # Convert 'Month' to datetime for sorting in calendar order
#     monthly_trends['Month'] = pd.to_datetime(monthly_trends['Month'], format='%Y-%m')
#     monthly_trends = monthly_trends.sort_values(by='Month')

#     most_leave_month = monthly_trends.loc[monthly_trends['Leave_Taken'].idxmax()]
#     least_leave_month = monthly_trends.loc[monthly_trends['Leave_Taken'].idxmin()]

#     most_leave_month_str = most_leave_month['Month'].strftime('%B %Y')
#     least_leave_month_str = least_leave_month['Month'].strftime('%B %Y')

#     st.line_chart(monthly_trends.set_index('Month'))

#     st.write(f"The month with the most leave taken is **{most_leave_month_str}** with **{most_leave_month['Leave_Taken']} days**.")
#     st.write(f"The month with the least leave taken is **{least_leave_month_str}** with **{least_leave_month['Leave_Taken']} days**.")












# import streamlit as st
# import pandas as pd

# # Employee leave dataset
# data = {
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
#     'Reason': ['Annual Leave', 'Personal Leave', 'Sick Leave', 'Casual Leave', 
#                'Annual Leave', 'Maternity Leave', 'Paternity Leave', 'Compassionate Leave', 'Study Leave']
# }
# df = pd.DataFrame(data)

# # Convert dates to datetime
# df['Start_Date'] = pd.to_datetime(df['Start_Date'])
# df['End_Date'] = pd.to_datetime(df['End_Date'])
# df['Month'] = df['Start_Date'].dt.strftime('%Y-%m')  # Extracting month in 'YYYY-MM' format

# # Multiselect widget for filtering leave types
# leave_types = df['Reason'].unique().tolist()
# selected_leave_types = st.multiselect('Select Leave Types to Display:', leave_types, default=leave_types)

# # Filter dataset based on selected leave types
# filtered_df = df[df['Reason'].isin(selected_leave_types)]

# # Display the filtered dataset
# st.write(f"Filtered Leave Data ({len(filtered_df)} records):")
# st.dataframe(filtered_df)

# # Metrics: Total leave taken and average leave remaining
# st.write(f"Total Leave Taken: {filtered_df['Leave_Taken'].sum()}")
# st.write(f"Average Leave Remaining: {filtered_df['Leave_Remaining'].mean():.2f}")

# # Employee statistics
# most_leave = df.loc[df['Leave_Taken'].idxmax()]
# least_leave = df.loc[df['Leave_Taken'].idxmin()]
# most_remaining = df.loc[df['Leave_Remaining'].idxmax()]
# least_remaining = df.loc[df['Leave_Remaining'].idxmin()]

# st.write("### Employee Statistics")
# st.write(f"Employee with the Most Leave Taken: {most_leave['Employee_Name']} ({most_leave['Leave_Taken']} days)")
# st.write(f"Employee with the Least Leave Taken: {least_leave['Employee_Name']} ({least_leave['Leave_Taken']} days)")
# st.write(f"Employee with the Most Leave Remaining: {most_remaining['Employee_Name']} ({most_remaining['Leave_Remaining']} days)")
# st.write(f"Employee with the Least Leave Remaining: {least_remaining['Employee_Name']} ({least_remaining['Leave_Remaining']} days)")

# # Count of employees by leave type
# leave_type_counts = filtered_df['Reason'].value_counts()
# st.write("### Count of Employees Taking Each Type of Leave")
# st.bar_chart(leave_type_counts)

# # Employee selection based on leave type
# selected_reason = st.selectbox("Select a Leave Type to View Employees:", leave_types)
# employees_on_leave = df[df['Reason'] == selected_reason][['Employee_ID', 'Employee_Name', 'Leave_Taken']]
# st.write(f"Employees on {selected_reason}:")
# st.dataframe(employees_on_leave)

# # Monthly trends for leaves taken
# monthly_trends = df.groupby('Month')['Leave_Taken'].sum().reset_index()

# # Convert 'Month' to datetime for sorting in calendar order
# monthly_trends['Month'] = pd.to_datetime(monthly_trends['Month'], format='%Y-%m')
# monthly_trends = monthly_trends.sort_values(by='Month')

# # Determine the month with the most and least leave taken
# most_leave_month = monthly_trends.loc[monthly_trends['Leave_Taken'].idxmax()]
# least_leave_month = monthly_trends.loc[monthly_trends['Leave_Taken'].idxmin()]

# # Convert month back to a readable string
# most_leave_month_str = most_leave_month['Month'].strftime('%B %Y')
# least_leave_month_str = least_leave_month['Month'].strftime('%B %Y')

# # Display monthly trends
# st.write("### Monthly Trends for Leaves Taken")
# st.line_chart(monthly_trends.set_index('Month'))

# # Display months with most and least leave taken
# st.write(f"The month with the most leave taken is **{most_leave_month_str}** with **{most_leave_month['Leave_Taken']} days**.")
# st.write(f"The month with the least leave taken is **{least_leave_month_str}** with **{least_leave_month['Leave_Taken']} days**.")















# import streamlit as st
# import pandas as pd

# # Employee leave dataset
# data = {
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
#     'Reason': ['Annual Leave', 'Personal Leave', 'Sick Leave', 'Casual Leave', 
#                'Annual Leave', 'Maternity Leave', 'Paternity Leave', 'Compassionate Leave', 'Study Leave']
# }
# df = pd.DataFrame(data)

# # Convert dates to datetime
# df['Start_Date'] = pd.to_datetime(df['Start_Date'])
# df['End_Date'] = pd.to_datetime(df['End_Date'])
# df['Month'] = df['Start_Date'].dt.strftime('%Y-%m')  # Extracting month in 'YYYY-MM' format

# # Multiselect widget for filtering leave types
# leave_types = df['Reason'].unique().tolist()
# selected_leave_types = st.multiselect('Select Leave Types to Display:', leave_types, default=leave_types)

# # Filter dataset based on selected leave types
# filtered_df = df[df['Reason'].isin(selected_leave_types)]

# # Display the filtered dataset
# st.write(f"Filtered Leave Data ({len(filtered_df)} records):")
# st.dataframe(filtered_df)

# # Metrics: Total leave taken and average leave remaining
# st.write(f"Total Leave Taken: {filtered_df['Leave_Taken'].sum()}")
# st.write(f"Average Leave Remaining: {filtered_df['Leave_Remaining'].mean():.2f}")

# # Employee statistics
# most_leave = df.loc[df['Leave_Taken'].idxmax()]
# least_leave = df.loc[df['Leave_Taken'].idxmin()]
# most_remaining = df.loc[df['Leave_Remaining'].idxmax()]
# least_remaining = df.loc[df['Leave_Remaining'].idxmin()]

# st.write("### Employee Statistics")
# st.write(f"Employee with the Most Leave Taken: {most_leave['Employee_Name']} ({most_leave['Leave_Taken']} days)")
# st.write(f"Employee with the Least Leave Taken: {least_leave['Employee_Name']} ({least_leave['Leave_Taken']} days)")
# st.write(f"Employee with the Most Leave Remaining: {most_remaining['Employee_Name']} ({most_remaining['Leave_Remaining']} days)")
# st.write(f"Employee with the Least Leave Remaining: {least_remaining['Employee_Name']} ({least_remaining['Leave_Remaining']} days)")

# # Count of employees by leave type
# leave_type_counts = filtered_df['Reason'].value_counts()
# st.write("### Count of Employees Taking Each Type of Leave")
# st.bar_chart(leave_type_counts)

# # Employee selection based on leave type
# selected_reason = st.selectbox("Select a Leave Type to View Employees:", leave_types)
# employees_on_leave = df[df['Reason'] == selected_reason][['Employee_ID', 'Employee_Name', 'Leave_Taken']]
# st.write(f"Employees on {selected_reason}:")
# st.dataframe(employees_on_leave)

# # Monthly trends for leaves taken
# monthly_trends = df.groupby('Month')['Leave_Taken'].sum().reset_index()

# # Determine the month with the most and least leave taken
# most_leave_month = monthly_trends.loc[monthly_trends['Leave_Taken'].idxmax()]
# least_leave_month = monthly_trends.loc[monthly_trends['Leave_Taken'].idxmin()]

# # Display monthly trends
# st.write("### Monthly Trends for Leaves Taken")
# st.line_chart(monthly_trends.set_index('Month'))

# # Display months with most and least leave taken
# st.write(f"The month with the most leave taken is **{most_leave_month['Month']}** with **{most_leave_month['Leave_Taken']} days**.")
# st.write(f"The month with the least leave taken is **{least_leave_month['Month']}** with **{least_leave_month['Leave_Taken']} days**.")









# import streamlit as st
# import pandas as pd

# # Employee leave dataset
# data = {
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
#     'Reason': ['Annual Leave', 'Personal Leave', 'Sick Leave', 'Casual Leave', 
#                'Annual Leave', 'Maternity Leave', 'Paternity Leave', 'Compassionate Leave', 'Study Leave']
# }
# df = pd.DataFrame(data)

# # Convert dates to datetime
# df['Start_Date'] = pd.to_datetime(df['Start_Date'])
# df['End_Date'] = pd.to_datetime(df['End_Date'])
# df['Month'] = df['Start_Date'].dt.strftime('%Y-%m')  # Extracting month in 'YYYY-MM' format

# # Multiselect widget for filtering leave types
# leave_types = df['Reason'].unique().tolist()
# selected_leave_types = st.multiselect('Select Leave Types to Display:', leave_types, default=leave_types)

# # Filter dataset based on selected leave types
# filtered_df = df[df['Reason'].isin(selected_leave_types)]

# # Display the filtered dataset
# st.write(f"Filtered Leave Data ({len(filtered_df)} records):")
# st.dataframe(filtered_df)

# # Metrics: Total leave taken and average leave remaining
# st.write(f"Total Leave Taken: {filtered_df['Leave_Taken'].sum()}")
# st.write(f"Average Leave Remaining: {filtered_df['Leave_Remaining'].mean():.2f}")

# # Employee statistics
# most_leave = df.loc[df['Leave_Taken'].idxmax()]
# least_leave = df.loc[df['Leave_Taken'].idxmin()]
# most_remaining = df.loc[df['Leave_Remaining'].idxmax()]
# least_remaining = df.loc[df['Leave_Remaining'].idxmin()]

# st.write("### Employee Statistics")
# st.write(f"Employee with the Most Leave Taken: {most_leave['Employee_Name']} ({most_leave['Leave_Taken']} days)")
# st.write(f"Employee with the Least Leave Taken: {least_leave['Employee_Name']} ({least_leave['Leave_Taken']} days)")
# st.write(f"Employee with the Most Leave Remaining: {most_remaining['Employee_Name']} ({most_remaining['Leave_Remaining']} days)")
# st.write(f"Employee with the Least Leave Remaining: {least_remaining['Employee_Name']} ({least_remaining['Leave_Remaining']} days)")

# # Count of employees by leave type
# leave_type_counts = filtered_df['Reason'].value_counts()
# st.write("### Count of Employees Taking Each Type of Leave")
# st.bar_chart(leave_type_counts)

# # Employee selection based on leave type
# selected_reason = st.selectbox("Select a Leave Type to View Employees:", leave_types)
# employees_on_leave = df[df['Reason'] == selected_reason][['Employee_ID', 'Employee_Name', 'Leave_Taken']]
# st.write(f"Employees on {selected_reason}:")
# st.dataframe(employees_on_leave)

# # Monthly trends for leaves taken
# monthly_trends = df.groupby('Month')['Leave_Taken'].sum().reset_index()
# st.write("### Monthly Trends for Leaves Taken")
# st.line_chart(monthly_trends.set_index('Month'))

















# import streamlit as st
# import pandas as pd

# # Employee leave dataset
# data = {
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
#     'Reason': ['Annual Leave', 'Personal Leave', 'Sick Leave', 'Casual Leave', 
#                'Annual Leave', 'Maternity Leave', 'Paternity Leave', 'Compassionate Leave', 'Study Leave']
# }
# df = pd.DataFrame(data)

# # Multiselect widget for filtering leave types
# leave_types = df['Reason'].unique().tolist()
# selected_leave_types = st.multiselect('Select Leave Types to Display:', leave_types, default=leave_types)

# # Filter dataset based on selected leave types
# filtered_df = df[df['Reason'].isin(selected_leave_types)]

# # Display the filtered dataset
# st.write(f"Filtered Leave Data ({len(filtered_df)} records):")
# st.dataframe(filtered_df)

# # Metrics: Total leave taken and average leave remaining
# st.write(f"Total Leave Taken: {filtered_df['Leave_Taken'].sum()}")
# st.write(f"Average Leave Remaining: {filtered_df['Leave_Remaining'].mean():.2f}")

# # Count of employees by leave type
# leave_type_counts = filtered_df['Reason'].value_counts()
# st.write("Count of Employees Taking Each Type of Leave:")
# st.bar_chart(leave_type_counts)

# # Employee selection based on leave type
# selected_reason = st.selectbox("Select a Leave Type to View Employees:", leave_types)
# employees_on_leave = df[df['Reason'] == selected_reason][['Employee_ID', 'Employee_Name', 'Leave_Taken']]
# st.write(f"Employees on {selected_reason}:")
# st.dataframe(employees_on_leave)










# import streamlit as st
# import pandas as pd

# # Employee leave dataset
# data = {
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
#     'Reason': ['Annual Leave', 'Personal Leave', 'Sick Leave', 'Casual Leave', 
#                'Annual Leave', 'Maternity Leave', 'Paternity Leave', 'Compassionate Leave', 'Study Leave']
# }
# df = pd.DataFrame(data)

# # Multiselect widget for filtering leave types
# leave_types = df['Reason'].unique().tolist()
# selected_leave_types = st.multiselect('Select Leave Types to Display:', leave_types, default=leave_types)

# # Filter dataset based on selected leave types
# filtered_df = df[df['Reason'].isin(selected_leave_types)]

# # Display the filtered dataset
# st.write(f"Filtered Leave Data ({len(filtered_df)} records):")
# st.dataframe(filtered_df)

# # Optional: Provide additional metrics
# st.write(f"Total Leave Taken: {filtered_df['Leave_Taken'].sum()}")
# st.write(f"Average Leave Remaining: {filtered_df['Leave_Remaining'].mean():.2f}")



















# import streamlit as st
# import pandas as pd

# # Employee leave dataset
# data = {
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
#     'Reason': ['Annual Leave', 'Personal Leave', 'Sick Leave', 'Casual Leave', 
#                'Annual Leave', 'Maternity Leave', 'Paternity Leave', 'Compassionate Leave', 'Study Leave']
# }
# df = pd.DataFrame(data)

# # Convert Start_Date and End_Date to datetime
# df['Start_Date'] = pd.to_datetime(df['Start_Date'])
# df['End_Date'] = pd.to_datetime(df['End_Date'])

# # Function to count weekdays between two dates
# def count_weekdays(start, end):
#     return pd.date_range(start, end, freq='B').size

# # Add a new column for weekdays taken
# df['Weekdays_Taken'] = df.apply(lambda row: count_weekdays(row['Start_Date'], row['End_Date']), axis=1)

# # Streamlit app
# st.title("Employee Leave Tracker")

# # Multiselect widget for filtering leave types
# leave_types = df['Reason'].unique().tolist()
# selected_leave_types = st.multiselect('Select Leave Types to Display:', leave_types, default=leave_types)

# # Date inputs for filtering
# start_date = st.date_input('Start Date:', value=pd.to_datetime('2024-01-01')).strftime('%Y-%m-%d')
# end_date = st.date_input('End Date:', value=pd.to_datetime('2024-12-31')).strftime('%Y-%m-%d')

# # Convert inputs to datetime
# start_date = pd.to_datetime(start_date)
# end_date = pd.to_datetime(end_date)

# # Filter dataset based on selected leave types and date range
# filtered_df = df[
#     (df['Reason'].isin(selected_leave_types)) &
#     (df['Start_Date'] >= start_date) &
#     (df['End_Date'] <= end_date)
# ]

# # Display the filtered dataset
# st.write(f"Filtered Leave Data ({len(filtered_df)} records):")
# st.dataframe(filtered_df)

# # Display metrics for weekdays
# st.write(f"Total Weekdays Taken: {filtered_df['Weekdays_Taken'].sum()}")
# st.write(f"Average Leave Remaining: {filtered_df['Leave_Remaining'].mean():.2f}")













# import streamlit as st
# import pandas as pd

# # Employee leave dataset
# data = {
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
#     'Reason': ['Annual Leave', 'Personal Leave', 'Sick Leave', 'Casual Leave', 
#                'Annual Leave', 'Maternity Leave', 'Paternity Leave', 'Compassionate Leave', 'Study Leave']
# }
# df = pd.DataFrame(data)

# # Convert Start_Date and End_Date to datetime
# df['Start_Date'] = pd.to_datetime(df['Start_Date'])
# df['End_Date'] = pd.to_datetime(df['End_Date'])

# # Multiselect widget for filtering leave types
# leave_types = df['Reason'].unique().tolist()
# selected_leave_types = st.multiselect('Select Leave Types to Display:', leave_types, default=leave_types)

# # Filter dataset based on selected leave types
# filtered_df = df[df['Reason'].isin(selected_leave_types)]

# # Add date input widgets for interactive date range filtering
# start_date = st.date_input('Start Date:', value=pd.to_datetime('2024-01-01'))
# end_date = st.date_input('End Date:', value=pd.to_datetime('2024-12-31'))

# # Apply date range filtering
# filtered_df = filtered_df[(filtered_df['Start_Date'] >= pd.to_datetime(start_date)) &
#                           (filtered_df['End_Date'] <= pd.to_datetime(end_date))]

# # Display the filtered dataset
# st.write(f"Filtered Leave Data ({len(filtered_df)} records):")
# st.dataframe(filtered_df)

# # Display additional metrics
# if not filtered_df.empty:
#     st.write(f"Total Leave Taken: {filtered_df['Leave_Taken'].sum()}")
#     st.write(f"Average Leave Remaining: {filtered_df['Leave_Remaining'].mean():.2f}")
# else:
#     st.write("No records match the selected filters.")










# import pandas as pd
# import streamlit as st

# # Sample DataFrame
# data = {
#     "Assigned to": ["Fatima Adamu", "Aisha Bello", "Musa Ahmed", "Ibrahim Umar", "Maryam Yusuf", 
#                     "Aliyu Abubakar", "Zainab Hassan", "Khadija Mohammed", "Umar Sani", "Abdul Rahman"],
#     "Combined Category": ["Deposit Betslip Complaints Agent Payment Issue", 
#                           "Withdrawal Deposit Issue Agent Miscommunication", 
#                           "Pending Deposit Withdrawal Delay Agent Incorrect Info", 
#                           "Deposit Bonus Complaints Agent Bonus Dispute", 
#                           "Withdrawal Account Locked Agent Account Error", 
#                           "Pending Deposit Delayed Payout Agent Unresponsiveness", 
#                           "Deposit Login Issue Agent Fraud", 
#                           "Withdrawal Unsettled Bet Agent Delay", 
#                           "Deposit Casino Issue Agent Mismanagement", 
#                           "Pending Deposit Virtual Bet Issue Agent Error"],
#     "Status": ["Resolved", "Pending", "Escalated", "In Progress", "Opened", 
#                "Resolved", "Pending", "Escalated", "In Progress", "Opened"]
# }

# df = pd.DataFrame(data)

# # Streamlit app title
# st.title("Customer Complaints Management System")

# # Sidebar filters
# st.sidebar.header("Filter Complaints")

# # Filter by Assigned To
# assigned_to = st.sidebar.multiselect(
#     "Filter by Assigned To:", options=df["Assigned to"].unique(), default=df["Assigned to"].unique()
# )

# # Filter by Status
# status = st.sidebar.multiselect(
#     "Filter by Status:", options=df["Status"].unique(), default=df["Status"].unique()
# )

# # Apply filters
# filtered_df = df[
#     (df["Assigned to"].isin(assigned_to)) &
#     (df["Status"].isin(status))
# ]

# # Display the filtered dataset
# st.header("Filtered Complaints")
# st.dataframe(filtered_df)

# # Option to download filtered data
# @st.cache_data
# def convert_df_to_csv(dataframe):
#     return dataframe.to_csv(index=False).encode('utf-8')

# csv = convert_df_to_csv(filtered_df)
# st.download_button(
#     label="Download Filtered Data as CSV",
#     data=csv,
#     file_name="filtered_complaints.csv",
#     mime="text/csv"
# )

# # Add summary of case statuses
# st.header("Summary of Case Statuses")

# # Group by Assigned To and Status, and count cases
# employee_status_counts = df.groupby(["Assigned to", "Status"]).size().unstack(fill_value=0)

# # Display the results
# st.subheader("Employee Case Counts by Status")
# st.dataframe(employee_status_counts)

# # Identify the employee with the highest cases for each status
# st.subheader("Top Employee by Case Status")
# top_employees = employee_status_counts.idxmax()
# for status, employee in top_employees.items():
#     st.write(f"*{status}*: {employee} with {employee_status_counts.loc[employee, status]} cases")

# # Count rows with "deposit" in Combined Category (case-insensitive)
# keyword = "deposit"
# filtered_deposit_df = df[df['Combined Category'].str.contains(keyword, case=False, na=False)]
# total_deposit_rows = len(filtered_deposit_df)

# # Display deposit-related metrics
# st.header("Deposit-Related Metrics")
# st.subheader(f"Total Rows Containing '{keyword}': {total_deposit_rows}")
# st.dataframe(filtered_deposit_df)






# import pandas as pd
# import streamlit as st

# # Upload customer data
# uploaded_file = st.file_uploader("Upload a CSV file", type="csv")

# if uploaded_file:
#     data = pd.read_csv(uploaded_file)
#     column = st.selectbox(
#         'Select a column to analyze:',
#         data.columns
#     )
#     st.write(f'Analyzing data from column: {column}')
#     st.write(data[column].describe())









# import streamlit as st

# continent = st.selectbox(
#     'Select a continent:',
#     ['Africa', 'Asia', 'Europe']
# )

# if continent == 'Africa':
#     country = st.selectbox(
#         'Select a country:',
#         ['Nigeria', 'Kenya', 'South Africa']
#     )
# elif continent == 'Asia':
#     country = st.selectbox(
#         'Select a country:',
#         ['India', 'China', 'Japan']
#     )
# else:
#     country = st.selectbox(
#         'Select a country:',
#         ['France', 'Germany', 'Spain']
#     )

# st.write(f'You selected {continent} and {country}.')










# import streamlit as st

# choice = st.selectbox(
#     'Select an action:',
#     ['Add', 'Subtract', 'Multiply', 'Divide']
# )

# if choice == 'Add':
#     st.write('You chose to add numbers.')
# elif choice == 'Subtract':
#     st.write('You chose to subtract numbers.')
# elif choice == 'Multiply':
#     st.write('You chose to multiply numbers.')
# else:
#     st.write('You chose to divide numbers.')






# import streamlit as st

# from enum import Enum

# class Fruit(Enum):
#     APPLE = "Apple"
#     BANANA = "Banana"
#     CHERRY = "Cherry"

# # Display Enum members
# selected_fruit = st.selectbox(
#     'Pick a fruit:',
#     list(Fruit)  # Convert Enum to list
# )

# st.write(f'You selected: {selected_fruit.value}')







# import streamlit as st


# #adding a selectbox

# choice = st.selectbox(

#     'Select the items you want?',

#     ('Pen','Pencil','Eraser','Sharpener','Notebook'))



# #displaying the selected option

# st.write('You have selected:', choice)











# import streamlit as st
# import pandas as pd

# # Load the dataset
# data_path = "customer_complaints_data.csv"
# df = pd.read_csv(data_path)

# # Streamlit app title
# st.title("Customer Complaints Management")

# # Sidebar filters
# st.sidebar.header("Filter Complaints")

# # Filter by Assigned To
# assigned_to = st.sidebar.multiselect(
#     "Filter by Assigned To:", options=df["Assigned to"].unique(), default=df["Assigned to"].unique()
# )

# # Filter by Status
# status = st.sidebar.multiselect(
#     "Filter by Status:", options=df["Status"].unique(), default=df["Status"].unique()
# )

# # Filter by Online Users
# online_users = st.sidebar.multiselect(
#     "Filter by Online Users:", options=df["Online Users"].unique(), default=df["Online Users"].unique()
# )

# # Filter by Agent Complaint
# agent_complaint = st.sidebar.multiselect(
#     "Filter by Agent Complaint:", options=df["Agent Complaint"].unique(), default=df["Agent Complaint"].unique()
# )

# # Apply filters
# filtered_df = df[
#     (df["Assigned to"].isin(assigned_to)) &
#     (df["Status"].isin(status)) &
#     (df["Online Users"].isin(online_users)) &
#     (df["Agent Complaint"].isin(agent_complaint))
# ]

# # Display the filtered dataset
# st.header("Filtered Complaints")
# st.dataframe(filtered_df)

# # Option to download filtered data
# @st.cache_data
# def convert_df_to_csv(dataframe):
#     return dataframe.to_csv(index=False).encode('utf-8')

# csv = convert_df_to_csv(filtered_df)
# st.download_button(
#     label="Download Filtered Data as CSV",
#     data=csv,
#     file_name="filtered_complaints.csv",
#     mime="text/csv"
# )

# # Add summary of case statuses
# st.header("Summary of Case Statuses")

# # Calculate the counts for each status
# status_counts = df["Status"].value_counts()

# open_cases = status_counts.get("Opened", 0)
# inprogress_cases = status_counts.get("In Progress", 0)
# resolved_cases = status_counts.get("Resolved", 0)
# escalated_cases = status_counts.get("Escalated", 0)

# # Display the results
# st.write(f"**Open Cases:** {open_cases}")
# st.write(f"**In-Progress Cases:** {inprogress_cases}")
# st.write(f"**Resolved Cases:** {resolved_cases}")
# st.write(f"**Escalated Cases:** {escalated_cases}")



















# import streamlit as st
# import pandas as pd

# # Employee leave dataset
# data = {
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
#     'Reason': ['Annual Leave', 'Personal Leave', 'Sick Leave', 'Casual Leave', 
#                'Annual Leave', 'Maternity Leave', 'Paternity Leave', 'Compassionate Leave', 'Study Leave']
# }
# df = pd.DataFrame(data)

# # Convert date columns to datetime format
# df['Start_Date'] = pd.to_datetime(df['Start_Date'])
# df['End_Date'] = pd.to_datetime(df['End_Date'])

# # Expander for grouped data
# with st.expander("Group Data by Leave Type"):
#     # Group data by leave type
#     grouped_data = df.groupby('Reason').agg(
#         Total_Leave_Taken=('Leave_Taken', 'sum'),
#         Total_Leave_Remaining=('Leave_Remaining', 'sum'),
#         Average_Leave_Remaining=('Leave_Remaining', 'mean'),
#         Number_of_Employees=('Employee_ID', 'count')
#     ).reset_index()

#     # Multiselect to filter leave types
#     leave_types = grouped_data['Reason'].unique().tolist()
#     selected_leave_types = st.multiselect(
#         'Select Leave Types to Display:', 
#         leave_types, 
#         default=leave_types
#     )

#     # Filter grouped data
#     filtered_grouped_data = grouped_data[grouped_data['Reason'].isin(selected_leave_types)]

#     # Display grouped data
#     st.write("Grouped Data by Leave Type:")
#     st.dataframe(filtered_grouped_data)

#     # Option to export grouped data
#     csv = filtered_grouped_data.to_csv(index=False).encode('utf-8')
#     st.download_button(
#         label="Download Grouped Data as CSV",
#         data=csv,
#         file_name="grouped_leave_data.csv",
#         mime="text/csv"
#     )


















# import streamlit as st
# import pandas as pd

# # Employee leave dataset
# data = {
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
#     'Reason': ['Annual Leave', 'Personal Leave', 'Sick Leave', 'Casual Leave', 
#                'Annual Leave', 'Maternity Leave', 'Paternity Leave', 'Compassionate Leave', 'Study Leave']
# }
# df = pd.DataFrame(data)

# # Convert date columns to datetime format
# df['Start_Date'] = pd.to_datetime(df['Start_Date'])
# df['End_Date'] = pd.to_datetime(df['End_Date'])

# # Expander for filtering options
# with st.expander("Filter Leave Types and View Data"):
#     leave_types = df['Reason'].unique().tolist()
#     selected_leave_types = st.multiselect(
#         'Select Leave Types to Display:', 
#         leave_types, 
#         default=leave_types
#     )

#     # Date range filter
#     start_date = pd.to_datetime(st.date_input('Start Date:', value=pd.to_datetime('2024-01-01')))
#     end_date = pd.to_datetime(st.date_input('End Date:', value=pd.to_datetime('2024-12-31')))

#     # Filter dataset
#     filtered_df = df[
#         (df['Reason'].isin(selected_leave_types)) & 
#         (df['Start_Date'] >= start_date) & 
#         (df['End_Date'] <= end_date)
#     ]

#     # Additional metrics
#     total_leave_taken = filtered_df['Leave_Taken'].sum()
#     average_leave_remaining = filtered_df['Leave_Remaining'].mean()

# # Display filtered data and metrics outside the expander
# st.write(f"Filtered Leave Data ({len(filtered_df)} records):")
# st.dataframe(filtered_df)

# st.write(f"**Total Leave Taken:** {total_leave_taken}")
# st.write(f"**Average Leave Remaining:** {average_leave_remaining:.2f}")

# # Option to export filtered data
# csv = filtered_df.to_csv(index=False).encode('utf-8')
# st.download_button(
#     label="Download Filtered Data as CSV",
#     data=csv,
#     file_name="filtered_leave_data.csv",
#     mime="text/csv"
# )














# import streamlit as st
# import pandas as pd

# # Employee leave dataset
# data = {
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
#     'Reason': ['Annual Leave', 'Personal Leave', 'Sick Leave', 'Casual Leave', 
#                'Annual Leave', 'Maternity Leave', 'Paternity Leave', 'Compassionate Leave', 'Study Leave']
# }
# df = pd.DataFrame(data)

# # Multiselect widget for filtering leave types

# leave_types = df['Reason'].unique().tolist()
# selected_leave_types = st.multiselect('Select Leave Types to Display:', leave_types, default=leave_types)

# # Filter dataset based on selected leave types
# filtered_df = df[df['Reason'].isin(selected_leave_types)]

# # Display the filtered dataset
# st.write(f"Filtered Leave Data ({len(filtered_df)} records):")
# st.dataframe(filtered_df)

# # Optional: Provide additional metrics
# st.write(f"Total Leave Taken: {filtered_df['Leave_Taken'].sum()}")
# st.write(f"Average Leave Remaining: {filtered_df['Leave_Remaining'].mean():.2f}")














# import streamlit as st
# import pandas as pd

# # Sample leave dataset
# data = {
#     'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Ella'],
#     'Annual Leave': [10, 12, 15, 5, 8],
#     'Casual Leave': [5, 3, 2, 4, 6],
#     'Personal Leave': [2, 1, 3, 0, 2],
#     'Sick Leave': [8, 6, 10, 7, 9],
#     'Maternity Leave': [0, 0, 0, 0, 10],
#     'Paternity Leave': [5, 7, 0, 8, 0],
#     'Department': ['HR', 'IT', 'Finance', 'IT', 'HR']
# }
# df = pd.DataFrame(data)

# # Multiselect to filter leave types
# leave_types = ['Annual Leave', 'Casual Leave', 'Personal Leave', 'Sick Leave', 'Maternity Leave', 'Paternity Leave']
# selected_leave_types = st.multiselect('Select Leave Types to Display:', leave_types, default=leave_types)

# # Include selected leave types in the display
# selected_columns = ['Name', 'Department'] + selected_leave_types
# st.write('Selected Leave Data:', df[selected_columns])










# import streamlit as st
# import pandas as pd

# # Sample dataset
# data = {
#     'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Ella'],
#     'Department': ['HR', 'IT', 'Finance', 'IT', 'HR'],
#     'Age': [25, 30, 35, 40, 28]
# }
# df = pd.DataFrame(data)

# # Multiselect to filter rows
# departments = df['Department'].unique()
# ages = df['Age'].unique()

# selected_departments = st.multiselect('Select Department(s):', departments)
# selected_ages = st.multiselect('Select Age(s):', ages)

# # Filter DataFrame
# filtered_df = df[(df['Department'].isin(selected_departments)) & (df['Age'].isin(selected_ages))]

# st.write('Filtered Data:', filtered_df)





# import streamlit as st
# import pandas as pd

# # Sample dataset
# data = {
#     'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Ella'],
#     'Department': ['HR', 'IT', 'Finance', 'IT', 'HR'],
#     'Age': [25, 30, 35, 40, 28]
# }
# df = pd.DataFrame(data)

# # Multiselect to filter rows
# departments = df['Department'].unique()
# selected_departments = st.multiselect('Select Department(s):', departments)

# # Filter DataFrame
# filtered_df = df[df['Department'].isin(selected_departments)]

# # Display filtered data
# st.write('Filtered Data:', filtered_df)

# # Button to export data
# if not filtered_df.empty:
#     csv = filtered_df.to_csv(index=False).encode('utf-8')
#     st.download_button(
#         label='Download Filtered Data as CSV',
#         data=csv,
#         file_name='filtered_data.csv',
#         mime='text/csv'
#     )











# import streamlit as st
# import pandas as pd

# # Sample dataset
# data = {
#     'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Ella'],
#     'Department': ['HR', 'IT', 'Finance', 'IT', 'HR'],
#     'Salary': [50000, 60000, 70000, 80000, 55000]
# }
# df = pd.DataFrame(data)

# # Multiselect to choose a column for grouping
# columns = ['Department']
# selected_column = st.multiselect('Select Column for Grouping:', columns)

# if selected_column:
#     # Group by selected column and calculate the mean for numeric columns only
#     grouped_data = df.groupby(selected_column)[['Salary']].sum()
#     st.write('Aggregated Data:', grouped_data)






# import streamlit as st
# import pandas as pd

# # Sample dataset
# data = {
#     'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Ella'],
#     'Department': ['HR', 'IT', 'Finance', 'IT', 'HR'],
#     'Age': [25, 30, 35, 40, 28]
# }
# df = pd.DataFrame(data)

# # Multiselect to choose columns
# columns = df.columns
# selected_columns = st.multiselect('Select Columns to Display:', columns, default=list(columns))

# # Display selected columns
# st.write('Selected Columns Data:', df[selected_columns])








# import streamlit as st
# import pandas as pd

# # Sample dataset
# data = {
#     'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Ella'],
#     'Department': ['HR', 'IT', 'Finance', 'IT', 'HR'],
#     'Age': [25, 30, 35, 40, 28]
# }
# df = pd.DataFrame(data)

# # Multiselect to filter departments
# departments = df['Department'].unique()
# selected_departments = st.multiselect('Select Department(s):', departments)

# # Filter DataFrame
# filtered_df = df[df['Department'].isin(selected_departments)]

# st.write('Filtered Data:', filtered_df)











# import streamlit as st

# st.title('Select Your Course')

# # Radio buttons for course selection
# course = st.radio(
#     'Which course would you like to take?',
#     ('Introduction to Data Science', 'Machine Learning 101', 'Python Programming', 'Web Development', 'AI Fundamentals')
# )

# if course:
#     st.write(f'You selected: {course}')
#     st.success('Your course selection has been submitted!')









# import streamlit as st
# import pandas as pd
# from scipy.stats import zscore
# import calendar
# import matplotlib.pyplot as plt

# # Load the data
# file_path = 'employee_leave_data.txt'
# df = pd.read_csv(file_path)

# # Convert dates to datetime
# df['Start_Date'] = pd.to_datetime(df['Start_Date'])
# df['End_Date'] = pd.to_datetime(df['End_Date'])

# # Calculate leave duration
# df['Leave_Duration'] = (df['End_Date'] - df['Start_Date']).dt.days + 1

# # Additional columns
# df['Leave_Utilization_%'] = (df['Leave_Taken'] / df['Total_Annual_Leave']) * 100

# # Categorize leave duration
# def categorize_duration(days):
#     if days <= 5:
#         return 'Short'
#     elif 6 <= days <= 10:
#         return 'Medium'
#     else:
#         return 'Long'

# df['Duration_Category'] = df['Leave_Duration'].apply(categorize_duration)
# df['Month'] = df['Start_Date'].dt.month

# # Identify the month with the most leave requests
# most_leave_month = df['Month'].value_counts().idxmax()
# most_leave_month_name = calendar.month_name[most_leave_month]

# # Calculate leave remaining
# df['Leave_Remaining'] = df['Total_Annual_Leave'] - df['Leave_Taken']

# # Group by 'Reason', aggregate Leave_Taken, and concatenate Employee_Name
# grouped_leave_reason = (
#     df.groupby('Reason')
#     .agg(
#         Leave_Taken=('Leave_Taken', 'sum'),
#         Employee_Name=('Employee_Name', lambda x: ', '.join(x))
#     )
#     .reset_index()
#     .sort_values(by='Leave_Taken', ascending=False)
# )

# # Sidebar options
# st.sidebar.title("Navigation")
# options = st.sidebar.radio(
#     "Choose a section:",
#     [
#         "Overview",
#         "Leave Analysis",
#         "Top Employees",
#         "Leave Trends",
#         "Threshold Analysis",
#         "Outliers",
#         "Eligible Employees",
#         "Reason Analysis",
#     ],
# )

# # Sidebar filters
# st.sidebar.header("Filters")
# leave_threshold = st.sidebar.slider("Set Leave Threshold", min_value=0, max_value=30, value=25)
# show_data = st.sidebar.checkbox("Show Raw Data", value=False)

# # Main app content
# st.title("Employee Leave Analysis Dashboard")

# if options == "Overview":
#     st.header("Overview")
#     st.write(
#         f"The month with the most leave requests is **{most_leave_month_name}**, "
#         f"with employees taking an average leave of **{df['Leave_Taken'].mean():.2f} days**."
#     )
#     st.metric("Total Leave Taken (days)", df['Leave_Taken'].sum())
#     st.metric("Average Leave Taken (days)", f"{df['Leave_Taken'].mean():.2f}")

# elif options == "Leave Analysis":
#     st.header("Leave Analysis")
#     st.subheader("Employee Leave Summary")
#     employee_leave_summary = df[['Employee_Name', 'Leave_Taken', 'Leave_Remaining', 'Reason']]
#     st.table(employee_leave_summary)

# elif options == "Top Employees":
#     st.header("Top Employees")
#     st.subheader("Top 3 Employees Who Have Taken the Most Leave")
#     top_3_employees = df.nlargest(3, 'Leave_Taken')[['Employee_Name', 'Leave_Taken', 'Reason']]
#     st.table(top_3_employees)

#     st.subheader("Least 3 Employees Who Have Taken the Least Leave")
#     least_3_employees = df.nsmallest(3, 'Leave_Taken')[['Employee_Name', 'Leave_Taken', 'Reason']]
#     st.table(least_3_employees)

# elif options == "Leave Trends":
#     st.header("Leave Trends")
#     leave_trends = df.groupby('Month').size().reset_index(name='Leave_Count')
#     leave_trends['Month_Name'] = leave_trends['Month'].apply(lambda x: calendar.month_name[x])
#     st.subheader("Leave Trends by Month")
#     st.table(leave_trends[['Month_Name', 'Leave_Count']])

# elif options == "Threshold Analysis":
#     st.header("Threshold Analysis")
#     st.subheader(f"Employees Exceeding {leave_threshold} Days of Leave")
#     exceeding_threshold = df[df['Leave_Taken'] > leave_threshold][['Employee_Name', 'Leave_Taken', 'Reason']]
#     st.table(exceeding_threshold)

# elif options == "Outliers":
#     st.header("Outliers")
#     st.subheader("Leave Duration Outliers (Z-Score Method)")
#     df['Leave_Duration_Z'] = zscore(df['Leave_Duration'])
#     outliers = df[abs(df['Leave_Duration_Z']) > 2][['Employee_Name', 'Leave_Duration', 'Leave_Duration_Z']]
#     st.table(outliers)

# elif options == "Eligible Employees":
#     st.header("Eligible Employees for Leave")
#     eligible_for_leave = df[df['Leave_Remaining'] > 5][['Employee_Name', 'Leave_Remaining']]
#     st.table(eligible_for_leave)

# elif options == "Reason Analysis":
#     st.header("Reason Analysis")
#     st.subheader("Leave Taken by Reason")
#     st.table(grouped_leave_reason)

#     st.subheader("Graph: Leave Taken by Reason")
#     fig, ax = plt.subplots(figsize=(10, 6))
#     ax.barh(grouped_leave_reason['Reason'], grouped_leave_reason['Leave_Taken'], color='skyblue')
#     ax.set_xlabel("Total Leave Taken")
#     ax.set_ylabel("Reason")
#     ax.set_title("Leave Taken by Reason")
#     st.pyplot(fig)

# # Show raw data if selected
# if show_data:
#     st.sidebar.header("Raw Data")
#     st.dataframe(df)
















# import streamlit as st
# import pandas as pd
# from scipy.stats import zscore
# import calendar

# # Load the data
# file_path = 'employee_leave_data.txt'
# df = pd.read_csv(file_path)

# # Convert dates to datetime
# df['Start_Date'] = pd.to_datetime(df['Start_Date'])
# df['End_Date'] = pd.to_datetime(df['End_Date'])

# # Calculate leave duration
# df['Leave_Duration'] = (df['End_Date'] - df['Start_Date']).dt.days + 1

# # Additional columns
# df['Leave_Utilization_%'] = (df['Leave_Taken'] / df['Total_Annual_Leave']) * 100

# # Categorize leave duration
# def categorize_duration(days):
#     if days <= 5:
#         return 'Short'
#     elif 6 <= days <= 10:
#         return 'Medium'
#     else:
#         return 'Long'

# df['Duration_Category'] = df['Leave_Duration'].apply(categorize_duration)
# df['Month'] = df['Start_Date'].dt.month

# # Identify the month with the most leave requests
# most_leave_month = df['Month'].value_counts().idxmax()
# most_leave_month_name = calendar.month_name[most_leave_month]

# # Calculate leave remaining
# df['Leave_Remaining'] = df['Total_Annual_Leave'] - df['Leave_Taken']

# # Sidebar options
# st.sidebar.title("Navigation")
# options = st.sidebar.radio(
#     "Choose a section:",
#     [
#         "Overview",
#         "Leave Analysis",
#         "Top Employees",
#         "Leave Trends",
#         "Threshold Analysis",
#         "Outliers",
#         "Eligible Employees",
#     ],
# )

# # Sidebar filters
# st.sidebar.header("Filters")
# leave_threshold = st.sidebar.slider("Set Leave Threshold", min_value=0, max_value=30, value=25)
# show_data = st.sidebar.checkbox("Show Raw Data", value=False)

# # Main app content
# st.title("Employee Leave Analysis Dashboard")

# if options == "Overview":
#     st.header("Overview")
#     st.write(
#         f"The month with the most leave requests is **{most_leave_month_name}**, "
#         f"with employees taking an average leave of **{df['Leave_Taken'].mean():.2f} days**."
#     )
#     st.metric("Total Leave Taken (days)", df['Leave_Taken'].sum())
#     st.metric("Average Leave Taken (days)", f"{df['Leave_Taken'].mean():.2f}")

# elif options == "Leave Analysis":
#     st.header("Leave Analysis")
#     leave_by_reason = df.groupby('Reason')['Leave_Taken'].sum().reset_index()
#     st.subheader("Leave Taken by Reason")
#     st.table(leave_by_reason)

# elif options == "Top Employees":
#     st.header("Top Employees")
#     st.subheader("Top 3 Employees Who Have Taken the Most Leave")
#     top_3_employees = df.nlargest(3, 'Leave_Taken')[['Employee_Name', 'Leave_Taken', 'Reason']]
#     st.table(top_3_employees)

#     st.subheader("Least 3 Employees Who Have Taken the Least Leave")
#     least_3_employees = df.nsmallest(3, 'Leave_Taken')[['Employee_Name', 'Leave_Taken', 'Reason']]
#     st.table(least_3_employees)

# elif options == "Leave Trends":
#     st.header("Leave Trends")
#     leave_trends = df.groupby('Month').size().reset_index(name='Leave_Count')
#     leave_trends['Month_Name'] = leave_trends['Month'].apply(lambda x: calendar.month_name[x])
#     st.subheader("Leave Trends by Month")
#     st.table(leave_trends[['Month_Name', 'Leave_Count']])

# elif options == "Threshold Analysis":
#     st.header("Threshold Analysis")
#     st.subheader(f"Employees Exceeding {leave_threshold} Days of Leave")
#     exceeding_threshold = df[df['Leave_Taken'] > leave_threshold][['Employee_Name', 'Leave_Taken', 'Reason']]
#     st.table(exceeding_threshold)

# elif options == "Outliers":
#     st.header("Outliers")
#     st.subheader("Leave Duration Outliers (Z-Score Method)")
#     df['Leave_Duration_Z'] = zscore(df['Leave_Duration'])
#     outliers = df[abs(df['Leave_Duration_Z']) > 2][['Employee_Name', 'Leave_Duration', 'Leave_Duration_Z']]
#     st.table(outliers)

# elif options == "Eligible Employees":
#     st.header("Eligible Employees for Leave")
#     eligible_for_leave = df[df['Leave_Remaining'] > 5][['Employee_Name', 'Leave_Remaining']]
#     st.table(eligible_for_leave)

# # Show raw data if selected
# if show_data:
#     st.sidebar.header("Raw Data")
#     st.dataframe(df)






# import streamlit as st
# import pandas as pd

# st.title("Select Your Favorite Courses")

# # Step 1: Define a list of available courses
# courses = [
#     "Python for Data Science",
#     "Machine Learning",
#     "Deep Learning",
#     "Web Development with Django",
#     "Frontend Development with React",
#     "Cloud Computing with AWS",
#     "Blockchain Fundamentals",
#     "Cybersecurity Essentials"
# ]

# # Step 2: Add a sidebar for filtering courses
# st.sidebar.header("Filter Courses")
# categories = {
#     "Programming": ["Python for Data Science", "Web Development with Django", "Frontend Development with React"],
#     "AI/ML": ["Machine Learning", "Deep Learning"],
#     "Technology": ["Cloud Computing with AWS", "Blockchain Fundamentals", "Cybersecurity Essentials"]
# }

# # Sidebar multiselect for category filtering
# selected_categories = st.sidebar.multiselect("Select Course Categories:", options=categories.keys())

# # Filter courses based on selected categories
# filtered_courses = []
# for category in selected_categories:
#     filtered_courses.extend(categories[category])

# # Show filtered courses in the main area
# st.write("### Available Courses")
# if filtered_courses:
#     available_courses = [course for course in courses if course in filtered_courses]
# else:
#     available_courses = courses

# # Step 3: Create a checkbox for each filtered course
# selected_courses = []
# st.write("Check the courses you like:")
# for course in available_courses:
#     if st.checkbox(course):
#         selected_courses.append(course)

# # Step 4: Trigger actions based on selections
# if selected_courses:
#     st.success(f"You have selected the following courses: {', '.join(selected_courses)}")
    
#     # Example of triggering events
#     st.write("Here are some actions you can take next:")
#     for course in selected_courses:
#         if course == "Python for Data Science":
#             st.write("- Start with this beginner-friendly guide: [Python Basics](https://docs.python.org/3/tutorial/)")
#         elif course == "Machine Learning":
#             st.write("- Explore this ML resource: [Machine Learning Crash Course](https://developers.google.com/machine-learning/crash-course)")
#         elif course == "Deep Learning":
#             st.write("- Begin with the famous Deep Learning book by Ian Goodfellow.")
#         elif course == "Web Development with Django":
#             st.write("- Build your first Django project with this [official tutorial](https://docs.djangoproject.com/en/4.0/intro/tutorial01/).")
#         elif course == "Frontend Development with React":
#             st.write("- Get started with React here: [React Docs](https://reactjs.org/docs/getting-started.html).")
#         elif course == "Cloud Computing with AWS":
#             st.write("- Learn AWS basics here: [AWS Training](https://aws.amazon.com/training/).")
#         elif course == "Blockchain Fundamentals":
#             st.write("- Dive into blockchain basics with this free course: [Blockchain Basics](https://www.coursera.org/learn/blockchain-basics).")
#         elif course == "Cybersecurity Essentials":
#             st.write("- Start protecting systems with this guide: [Cybersecurity Essentials](https://www.cisa.gov/cyber-essentials).")

#     # Step 5: Prepare a downloadable file
#     file_data = pd.DataFrame({"Selected Courses": selected_courses})
#     csv_data = file_data.to_csv(index=False)
#     st.download_button(
#         label="Download Your Course Selection",
#         data=csv_data,
#         file_name="selected_courses.csv",
#         mime="text/csv"
#     )
# else:
#     st.info("Please select at least one course to proceed.")











# import pandas as pd
# import streamlit as st

# # Configure the Streamlit page
# st.set_page_config(page_title="Dashboard", page_icon="🌍", layout="wide")

# # Set the header
# st.header("ANALYTICAL PROCESSING, KPI, TRENDS & PREDICTIONS")

# # Create the dataset
# data = {
#     "Policy": [100242, 100314, 100359, 100315, 100385, 100388, 100358, 100264, 100265, 100357, 100399, 100329, 100429, 100441],
#     "Expiry": ["2-Jan-21", "2-Jan-21", "2-Jan-21", "3-Jan-21", "3-Jan-21", "4-Jan-21", "5-Jan-21", "7-Jan-21", "7-Jan-21", "8-Jan-21", "8-Jan-21", "9-Jan-21", "9-Jan-21", "10-Jan-21"],
#     "Location": ["Urban", "Urban", "Rural", "Urban", "Urban", "Urban", "Urban", "Rural", "Urban", "Urban", "Urban", "Urban", "Urban", "Urban"],
#     "State": ["Dodoma", "Kigoma", "Dodoma", "Dodoma", "Iringa", "Kigoma", "Dodoma", "Dodoma", "Dodoma", "Dodoma", "Dodoma", "Kigoma", "Dar es Salaam", "Kigoma"],
#     "Region": ["East", "East", "Midwest", "East", "East", "Midwest", "Midwest", "East", "East", "East", "East", "East", "Midwest", "East"],
#     "Investment": [1617630, 8678500, 2052660, 17580000, 1925000, 12934500, 928300, 2219900, 14100000, 4762808, 13925190, 6350000, 4036000, 472800],
#     "Construction": ["Frame", "Fire Resist", "Frame", "Frame", "Masonry", "Frame", "Masonry", "Frame", "Frame", "Masonry", "Frame", "Frame", "Masonry", "Masonry"],
#     "BusinessType": ["Retail", "Apartment", "Farming", "Apartment", "Hospitality", "Apartment", "Office Bldg", "Farming", "Apartment", "Other", "Apartment", "Apartment", "Medical", "Retail"],
#     "Earthquake": ["N", "Y", "N", "Y", "N", "Y", "N", "N", "Y", "Y", "Y", "Y", "Y", "Y"],
#     "Flood": ["N", "Y", "N", "Y", "N", "Y", "N", "N", "Y", "Y", "Y", "Y", "Y", "Y"],
#     "Rating": [9.1, 9.6, 7.4, 8.4, 5.3, 4.1, 5.8, 8.0, 7.2, 5.9, 4.5, 6.8, 7.1, 8.2]
# }

# # Load the data into a pandas DataFrame
# df = pd.DataFrame(data)

# # Convert Expiry to datetime format
# df['Expiry'] = pd.to_datetime(df['Expiry'], format='%d-%b-%y')

# # Calculate KPIs
# total_investment = df["Investment"].sum()
# most_investment = df["Investment"].mode().iloc[0]
# average_investment = df["Investment"].mean()
# median_investment = df["Investment"].median()
# average_rating = df["Rating"].mean()

# # Display KPIs
# st.markdown("### Key Performance Indicators (KPIs)")
# col1, col2, col3, col4, col5 = st.columns(5)

# with col1:
#     st.metric("Sum TZS", f"{total_investment:,} 💰")

# with col2:
#     st.metric("Most Investment (Mode)", f"{most_investment:,} 💰")

# with col3:
#     st.metric("Average Investment", f"{average_investment:,.0f} 💰")

# with col4:
#     st.metric("Median Investment", f"{median_investment:,.0f} 💰")

# with col5:
#     st.metric("Average Rating", f"{average_rating:.2f} ⭐")

# # Detailed Analysis
# st.markdown("### Detailed Analysis")

# # Total Investment by Region
# investment_by_region = df.groupby("Region")["Investment"].sum()
# st.bar_chart(investment_by_region)

# # Number of Policies by Construction Type
# policies_by_construction = df["Construction"].value_counts()
# st.bar_chart(policies_by_construction)

# # Display the full dataset
# st.markdown("### Full Dataset")
# st.dataframe(df)

# # Filter policies with ratings above 8
# st.markdown("### Policies with Ratings Above 8")
# high_rating_policies = df[df["Rating"] > 8]
# st.dataframe(high_rating_policies)

# # Average rating by BusinessType
# st.markdown("### Average Rating by Business Type")
# avg_rating_by_businesstype = df.groupby("BusinessType")["Rating"].mean()
# st.write(avg_rating_by_businesstype)

# # Display coverage percentages
# earthquake_coverage = (df["Earthquake"] == "Y").mean() * 100
# flood_coverage = (df["Flood"] == "Y").mean() * 100

# st.markdown("### Coverage Percentages")
# st.write(f"Percentage of Policies with Earthquake Coverage: {earthquake_coverage:.2f}%")
# st.write(f"Percentage of Policies with Flood Coverage: {flood_coverage:.2f}%")





# import streamlit as st
# import pandas as pd

# # Sample dataset
# data = {
#     'Complaint_ID': [101, 102, 103, 104, 105],
#     'Category': ['Payment', 'Account Issue', 'Bet Settlement', 'Technical', 'Payment'],
#     'Customer_Name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
#     'Status': ['Resolved', 'Pending', 'Resolved', 'Escalated', 'Pending']
# }

# # Convert to DataFrame
# df = pd.DataFrame(data)

# # Multiselect widget
# categories = df['Category'].unique().tolist()
# selected_categories = st.multiselect('Select complaint categories:', categories)

# # Button to filter data
# if st.button('Filter Data'):
#     if selected_categories:
#         filtered_df = df[df['Category'].isin(selected_categories)]
#         st.success('Data filtered successfully!')
#         st.write('Filtered Data:')
#         st.dataframe(filtered_df)
#     else:
#         st.warning('No categories selected for filtering.')

# # Button to display summary
# if st.button('Show Summary'):
#     if selected_categories:
#         filtered_df = df[df['Category'].isin(selected_categories)]
#         summary = filtered_df.groupby('Category').size()
#         st.success('Summary generated successfully!')
#         st.write('Summary of Complaints by Category:')
#         st.write(summary)
#     else:
#         st.warning('Please select categories to generate a summary.')





# import streamlit as st
# import pandas as pd
# import plotly.express as px

# # Sample dataset
# data = {
#     'Complaint_ID': [101, 102, 103, 104, 105, 106],
#     'Category': ['Payment', 'Account Issue', 'Bet Settlement', 'Technical', 'Payment', 'Technical'],
#     'Customer_Name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve', 'Frank'],
#     'Status': ['Resolved', 'Pending', 'Resolved', 'Escalated', 'Pending', 'Resolved'],
#     'Resolution_Time': [5, 8, 4, 10, 7, 3]
# }

# # Convert to DataFrame
# df = pd.DataFrame(data)

# # Sidebar
# st.sidebar.header('Actions')
# categories = df['Category'].unique().tolist()
# selected_categories = st.sidebar.multiselect('Select complaint categories:', categories)
# filter_data = st.sidebar.button('Filter Data')
# show_summary = st.sidebar.button('Show Summary')
# show_chart = st.sidebar.button('Show Chart')

# # Main area
# st.title('Customer Complaints Dashboard')

# if filter_data:
#     if selected_categories:
#         filtered_df = df[df['Category'].isin(selected_categories)]
#         st.success('Data filtered successfully!')
#         st.write('Filtered Data:')
#         st.dataframe(filtered_df)
#     else:
#         st.warning('Please select at least one category to filter the data.')

# if show_summary:
#     if selected_categories:
#         filtered_df = df[df['Category'].isin(selected_categories)]
#         summary = filtered_df.groupby('Category').size()
#         st.success('Summary generated successfully!')
#         st.write('Summary of Complaints by Category:')
#         st.write(summary)
#     else:
#         st.warning('Please select categories to generate a summary.')

# if show_chart:
#     if selected_categories:
#         filtered_df = df[df['Category'].isin(selected_categories)]
#         if not filtered_df.empty:
#             fig = px.bar(
#                 filtered_df,
#                 x='Category',
#                 y='Resolution_Time',
#                 color='Status',
#                 title='Resolution Time by Complaint Category'
#             )
#             st.plotly_chart(fig)
#         else:
#             st.warning('No data available for the selected categories.')
#     else:
#         st.error('Please select at least one category to display the chart.')




















# import streamlit as st
# import pandas as pd
# import plotly.express as px

# # App Title
# st.title('Comprehensive Customer Complaints Dashboard')

# # Sidebar Widgets
# st.sidebar.header('Sidebar Controls')

# # File Uploader
# uploaded_file = st.sidebar.file_uploader("Upload your dataset", type=["csv", "xlsx"])
# if uploaded_file:
#     if uploaded_file.name.endswith('.csv'):
#         df = pd.read_csv(uploaded_file)
#     else:
#         df = pd.read_excel(uploaded_file)
# else:
#     # Default dataset
#     data = {
#         'Complaint_ID': [101, 102, 103, 104, 105, 106],
#         'Category': ['Payment', 'Account Issue', 'Bet Settlement', 'Technical', 'Payment', 'Technical'],
#         'Customer_Name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve', 'Frank'],
#         'Status': ['Resolved', 'Pending', 'Resolved', 'Escalated', 'Pending', 'Resolved'],
#         'Resolution_Time': [5, 8, 4, 10, 7, 3],
#         'Complaint_Date': pd.date_range(start='2023-01-01', periods=6, freq='D')
#     }
#     df = pd.DataFrame(data)

# # Checkbox
# if st.sidebar.checkbox("Show raw data"):
#     st.subheader("Raw Data")
#     st.dataframe(df)

# # Radio Button
# view_type = st.sidebar.radio("Select view type:", ("Overview", "Analysis"))

# # Text Input
# custom_message = st.sidebar.text_input("Enter a custom message:", value="Welcome to the dashboard!")

# # Slider
# days_filter = st.sidebar.slider("Filter complaints by days since filing:", 0, 10, 5)

# # Selectbox
# categories = df['Category'].unique().tolist()
# selected_category = st.sidebar.selectbox("Select a category to filter:", ["All"] + categories)

# # Create Columns
# col1, col2, col3 = st.columns(3)

# # Column Metrics
# with col1:
#     st.metric(label="Total Complaints", value=len(df))
# with col2:
#     st.metric(label="Resolved Complaints", value=len(df[df["Status"] == "Resolved"]))
# with col3:
#     st.metric(label="Pending Complaints", value=len(df[df["Status"] == "Pending"]))

# # Filter Data
# filtered_df = df[df['Resolution_Time'] <= days_filter]
# if selected_category != "All":
#     filtered_df = filtered_df[filtered_df["Category"] == selected_category]

# # Barchart
# st.subheader("Barchart: Complaints by Category")
# fig_bar = px.bar(filtered_df, x='Category', y='Resolution_Time', color='Status', title='Resolution Time by Category')
# st.plotly_chart(fig_bar)

# # Line Chart with Animation Frame
# st.subheader("Line Chart: Complaints Over Time")
# fig_line = px.line(
#     filtered_df,
#     x='Complaint_Date',
#     y='Resolution_Time',
#     color='Category',
#     animation_frame='Complaint_Date',
#     title='Resolution Time Over Days'
# )
# st.plotly_chart(fig_line)

# # File Upload Feedback
# if uploaded_file:
#     st.success(f"Dataset '{uploaded_file.name}' uploaded successfully!")

# # Take a Photo
# photo = st.camera_input("Take a photo for verification:")
# if photo:
#     st.image(photo, caption="Captured Photo")

# # Custom Message Display
# st.markdown(f"### {custom_message}")








# import streamlit as st
# import pandas as pd
# import plotly.express as px
# import time

# # App Title
# st.title('Comprehensive Customer Complaints Dashboard')

# # Sidebar Widgets
# st.sidebar.header('Sidebar Controls')

# # File Uploader
# uploaded_file = st.sidebar.file_uploader("Upload your dataset", type=["csv", "xlsx"])
# if uploaded_file:
#     with st.spinner("Loading dataset..."):
#         time.sleep(2)  # Simulating loading time
#         if uploaded_file.name.endswith('.csv'):
#             df = pd.read_csv(uploaded_file)
#         else:
#             df = pd.read_excel(uploaded_file)
#         st.sidebar.success("Dataset uploaded successfully!")
# else:
#     # Default dataset
#     data = {
#         'Complaint_ID': [101, 102, 103, 104, 105, 106],
#         'Category': ['Payment', 'Account Issue', 'Bet Settlement', 'Technical', 'Payment', 'Technical'],
#         'Customer_Name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve', 'Frank'],
#         'Status': ['Resolved', 'Pending', 'Resolved', 'Escalated', 'Pending', 'Resolved'],
#         'Resolution_Time': [5, 8, 4, 10, 7, 3],
#         'Complaint_Date': pd.date_range(start='2023-01-01', periods=6, freq='D')
#     }
#     df = pd.DataFrame(data)

# # Expander for Raw Data
# with st.expander("View Raw Data", expanded=False):
#     st.subheader("Raw Data")
#     st.dataframe(df)

# # Radio Button
# view_type = st.sidebar.radio("Select view type:", ("Overview", "Analysis"))

# # Text Input
# custom_message = st.sidebar.text_input("Enter a custom message:", value="Welcome to the dashboard!")

# # Slider
# days_filter = st.sidebar.slider("Filter complaints by days since filing:", 0, 10, 5)

# # Selectbox
# categories = df['Category'].unique().tolist()
# selected_category = st.sidebar.selectbox("Select a category to filter:", ["All"] + categories)

# # Progress Bar
# st.sidebar.text("Filtering Data...")
# progress = st.sidebar.progress(0)
# for percent in range(0, 101, 10):
#     time.sleep(0.05)  # Simulating work
#     progress.progress(percent)

# # Filter Data
# filtered_df = df[df['Resolution_Time'] <= days_filter]
# if selected_category != "All":
#     filtered_df = filtered_df[filtered_df["Category"] == selected_category]

# # Create Columns
# col1, col2, col3 = st.columns(3)

# # Column Metrics
# with col1:
#     st.metric(label="Total Complaints", value=len(df))
# with col2:
#     st.metric(label="Resolved Complaints", value=len(df[df["Status"] == "Resolved"]))
# with col3:
#     st.metric(label="Pending Complaints", value=len(df[df["Status"] == "Pending"]))

# # Spinner for Long Computations
# with st.spinner("Generating charts..."):
#     time.sleep(2)  # Simulating computation time

# # Barchart
# st.subheader("Barchart: Complaints by Category")
# fig_bar = px.bar(filtered_df, x='Category', y='Resolution_Time', color='Status', title='Resolution Time by Category')
# st.plotly_chart(fig_bar)

# # Line Chart with Animation Frame
# st.subheader("Line Chart: Complaints Over Time")
# fig_line = px.line(
#     filtered_df,
#     x='Complaint_Date',
#     y='Resolution_Time',
#     color='Category',
#     animation_frame='Complaint_Date',
#     title='Resolution Time Over Days'
# )
# st.plotly_chart(fig_line)

# # File Upload Feedback
# if uploaded_file:
#     st.success(f"Dataset '{uploaded_file.name}' uploaded successfully!")

# # Camera Input
# photo = st.camera_input("Take a photo for verification:")
# if photo:
#     st.image(photo, caption="Captured Photo")

# # Custom Message Display
# st.markdown(f"### {custom_message}")




















# import streamlit as st
# import pandas as pd
# import plotly.express as px
# import time

# # App Title
# st.title('Comprehensive Customer Complaints Dashboard')

# # Sidebar Widgets
# st.sidebar.header('Sidebar Controls')

# # File Uploader
# uploaded_file = st.sidebar.file_uploader("Upload your dataset", type=["csv", "xlsx"])
# if uploaded_file:
#     with st.spinner("Loading dataset..."):
#         time.sleep(2)  # Simulating loading time
#         if uploaded_file.name.endswith('.csv'):
#             df = pd.read_csv(uploaded_file)
#         else:
#             df = pd.read_excel(uploaded_file)
#         st.sidebar.success("Dataset uploaded successfully!")
# else:
#     # Default dataset
#     data = {
#         'Complaint_ID': [101, 102, 103, 104, 105, 106],
#         'Category': ['Payment', 'Account Issue', 'Bet Settlement', 'Technical', 'Payment', 'Technical'],
#         'Customer_Name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve', 'Frank'],
#         'Status': ['Resolved', 'Pending', 'Resolved', 'Escalated', 'Pending', 'Resolved'],
#         'Resolution_Time': [5, 8, 4, 10, 7, 3],
#         'Complaint_Date': pd.date_range(start='2023-01-01', periods=6, freq='D')
#     }
#     df = pd.DataFrame(data)

# # Expander for Raw Data
# with st.expander("View Raw Data", expanded=False):
#     st.subheader("Raw Data")
#     st.dataframe(df)

# # Radio Button
# view_type = st.sidebar.radio("Select view type:", ("Overview", "Analysis"))

# # Text Input
# custom_message = st.sidebar.text_input("Enter a custom message:", value="Welcome to the dashboard!")

# # Slider
# days_filter = st.sidebar.slider("Filter complaints by days since filing:", 0, 10, 5)

# # Selectbox
# categories = df['Category'].unique().tolist()
# selected_category = st.sidebar.selectbox("Select a category to filter:", ["All"] + categories)

# # Progress Bar
# st.sidebar.text("Filtering Data...")
# progress = st.sidebar.progress(0)
# for percent in range(0, 101, 10):
#     time.sleep(0.05)  # Simulating work
#     progress.progress(percent)

# # Filter Data
# filtered_df = df[df['Resolution_Time'] <= days_filter]
# if selected_category != "All":
#     filtered_df = filtered_df[filtered_df["Category"] == selected_category]

# # Create Columns
# col1, col2, col3, col4 = st.columns(4)

# # Column Metrics
# with col1:
#     st.metric(label="Total Complaints", value=len(df))
# with col2:
#     st.metric(label="Resolved Complaints", value=len(df[df["Status"] == "Resolved"]))
# with col3:
#     st.metric(label="Pending Complaints", value=len(df[df["Status"] == "Pending"]))
# with col4:
#     st.metric(label="Escalated Complaints", value=len(df[df["Status"] == "Escalated"]))

# # Spinner for Long Computations
# with st.spinner("Generating charts..."):
#     time.sleep(2)  # Simulating computation time

# # Barchart
# st.subheader("Barchart: Complaints by Category")
# fig_bar = px.bar(filtered_df, x='Category', y='Resolution_Time', color='Status', title='Resolution Time by Category')
# st.plotly_chart(fig_bar)

# # Line Chart with Animation Frame
# st.subheader("Line Chart: Complaints Over Time (Animation)")
# fig_line = px.line(
#     filtered_df,
#     x='Complaint_Date',
#     y='Resolution_Time',
#     color='Category',
#     animation_frame=filtered_df['Complaint_Date'].dt.strftime('%Y-%m-%d'),
#     title='Resolution Time Over Days'
# )
# st.plotly_chart(fig_line)

# # Donut Chart
# st.subheader("Donut Chart: Complaints by Status")
# fig_donut = px.pie(
#     df,
#     names='Status',
#     values='Resolution_Time',
#     hole=0.4,
#     title='Distribution of Complaints by Status'
# )
# st.plotly_chart(fig_donut)

# # File Upload Feedback
# if uploaded_file:
#     st.success(f"Dataset '{uploaded_file.name}' uploaded successfully!")

# # Camera Input
# photo = st.camera_input("Take a photo for verification:")
# if photo:
#     st.image(photo, caption="Captured Photo")

# # Custom Message Display
# st.markdown(f"### {custom_message}")















# import streamlit as st
# import pandas as pd
# import plotly.express as px
# import time

# # App Title
# st.title('Comprehensive Customer Complaints Dashboard')

# # Sidebar Widgets
# st.sidebar.header('Sidebar Controls')

# # File Uploader
# uploaded_file = st.sidebar.file_uploader("Upload your dataset", type=["csv", "xlsx"])
# if uploaded_file:
#     with st.spinner("Loading dataset..."):
#         time.sleep(2)  # Simulating loading time
#         if uploaded_file.name.endswith('.csv'):
#             df = pd.read_csv(uploaded_file)
#         else:
#             df = pd.read_excel(uploaded_file)
#         st.sidebar.success("Dataset uploaded successfully!")
# else:
#     # Default dataset
#     data = {
#         'Complaint_ID': [101, 102, 103, 104, 105, 106],
#         'Category': ['Payment', 'Account Issue', 'Bet Settlement', 'Technical', 'Payment', 'Technical'],
#         'Customer_Name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve', 'Frank'],
#         'Status': ['Resolved', 'Pending', 'Resolved', 'Escalated', 'Pending', 'Resolved'],
#         'Resolution_Time': [5, 8, 4, 10, 7, 3],
#         'Complaint_Date': pd.date_range(start='2023-01-01', periods=6, freq='D')
#     }
#     df = pd.DataFrame(data)

# # Sidebar Filters
# st.sidebar.subheader("Filter Complaints")
# days_filter = st.sidebar.slider("Filter complaints by days since filing:", 0, 10, 5)
# categories = df['Category'].unique().tolist()
# selected_category = st.sidebar.selectbox("Select a category to filter:", ["All"] + categories)

# # Sidebar Button
# apply_sidebar_filter = st.sidebar.button("Apply Sidebar Filter")

# # Filter using Sidebar Input
# if apply_sidebar_filter:
#     filtered_df = df[df['Resolution_Time'] <= days_filter]
#     if selected_category != "All":
#         filtered_df = filtered_df[filtered_df["Category"] == selected_category]
#     st.sidebar.success("Sidebar filter applied successfully!")
# else:
#     filtered_df = df  # Default to original dataset if no filtering applied

# # Expander for Raw Data
# with st.expander("View Raw Data", expanded=False):
#     st.subheader("Raw Data")
#     st.dataframe(df)

# # Selectbox for Validation
# selected_status = st.selectbox("Filter by complaint status (validation):", ["All"] + df['Status'].unique().tolist())

# # Validate Filter
# if selected_status != "All":
#     validated_df = filtered_df[filtered_df["Status"] == selected_status]
# else:
#     validated_df = filtered_df

# if validated_df.empty:
#     st.warning("No data available for the selected filters!")
# else:
#     st.success("Data successfully filtered based on selection!")

# # Button to Display Final Filtered Data
# if st.button("Show Filtered Data"):
#     st.subheader("Filtered Data")
#     st.dataframe(validated_df)

# # Metrics Section
# col1, col2, col3, col4 = st.columns(4)
# with col1:
#     st.metric(label="Total Complaints", value=len(df))
# with col2:
#     st.metric(label="Resolved Complaints", value=len(df[df["Status"] == "Resolved"]))
# with col3:
#     st.metric(label="Pending Complaints", value=len(df[df["Status"] == "Pending"]))
# with col4:
#     st.metric(label="Escalated Complaints", value=len(df[df["Status"] == "Escalated"]))

# # Spinner for Long Computations
# with st.spinner("Generating charts..."):
#     time.sleep(2)

# # Barchart
# st.subheader("Barchart: Complaints by Category")
# fig_bar = px.bar(validated_df, x='Category', y='Resolution_Time', color='Status', title='Resolution Time by Category')
# st.plotly_chart(fig_bar)

# # Line Chart with Animation Frame
# st.subheader("Line Chart: Complaints Over Time (Animation)")
# fig_line = px.line(
#     validated_df,
#     x='Complaint_Date',
#     y='Resolution_Time',
#     color='Category',
#     animation_frame=validated_df['Complaint_Date'].dt.strftime('%Y-%m-%d'),
#     title='Resolution Time Over Days'
# )
# st.plotly_chart(fig_line)

# # Donut Chart
# st.subheader("Donut Chart: Complaints by Status")
# fig_donut = px.pie(
#     validated_df,
#     names='Status',
#     values='Resolution_Time',
#     hole=0.4,
#     title='Distribution of Complaints by Status'
# )
# st.plotly_chart(fig_donut)

# # Camera Input
# photo = st.camera_input("Take a photo for verification:")
# if photo:
#     st.image(photo, caption="Captured Photo")



















# import streamlit as st
# import pandas as pd
# import plotly.express as px
# import time

# # App Title
# st.title("Bet9ja - Customer Complaint Resolution Dashboard")

# # Sidebar Elements
# st.sidebar.header("Controls and Filters")

# # File Uploader
# uploaded_file = st.sidebar.file_uploader("Upload Complaints Dataset", type=["csv", "xlsx"])
# if uploaded_file:
#     with st.spinner("Loading dataset..."):
#         time.sleep(2)  # Simulating loading time
#         if uploaded_file.name.endswith(".csv"):
#             df = pd.read_csv(uploaded_file)
#         else:
#             df = pd.read_excel(uploaded_file)
#         st.sidebar.success("Dataset uploaded successfully!")
# else:
#     # Default Dataset
#     data = {
#         "Complaint_ID": [101, 102, 103, 104, 105, 106],
#         "Category": ["Payment", "Account Issue", "Bet Settlement", "Technical", "Payment", "Technical"],
#         "Customer_Name": ["Alice", "Bob", "Charlie", "David", "Eve", "Frank"],
#         "Status": ["Resolved", "Pending", "Resolved", "Escalated", "Pending", "Resolved"],
#         "Resolution_Time": [5, 8, 4, 10, 7, 3],
#         "Complaint_Date": pd.date_range(start="2023-01-01", periods=6, freq="D"),
#     }
#     df = pd.DataFrame(data)

# # Sidebar Filters
# st.sidebar.subheader("Filter Complaints")
# days_filter = st.sidebar.slider("Days Since Filing:", 0, 10, 5)
# categories = df["Category"].unique().tolist()
# selected_category = st.sidebar.selectbox("Category Filter:", ["All"] + categories)
# statuses = df["Status"].unique().tolist()
# selected_status = st.sidebar.radio("Status Filter:", ["All"] + statuses)

# # Button to Apply Filters
# if st.sidebar.button("Apply Filters"):
#     filtered_df = df[df["Resolution_Time"] <= days_filter]
#     if selected_category != "All":
#         filtered_df = filtered_df[filtered_df["Category"] == selected_category]
#     if selected_status != "All":
#         filtered_df = filtered_df[filtered_df["Status"] == selected_status]
# else:
#     filtered_df = df

# # Expander to View Raw Data
# with st.expander("View Raw Dataset", expanded=False):
#     st.subheader("Raw Data")
#     st.dataframe(df)

# # Input Elements
# st.subheader("Add New Complaint Record")
# with st.form("complaint_form"):
#     customer_name = st.text_input("Customer Name:")
#     complaint_category = st.selectbox("Complaint Category:", categories)
#     complaint_status = st.radio("Complaint Status:", ["Pending", "Resolved", "Escalated"])
#     resolution_time = st.slider("Resolution Time (days):", 0, 15, 5)
#     submitted = st.form_submit_button("Add Complaint")
#     if submitted:
#         st.success(f"Complaint added for {customer_name} in {complaint_category} category.")

# # Progress Bar
# with st.spinner("Resolving complaints..."):
#     for i in range(101):
#         time.sleep(0.02)  # Simulating resolution process
#         st.progress(i)

# # Metrics Section
# col1, col2, col3, col4 = st.columns(4)
# with col1:
#     st.metric(label="Total Complaints", value=len(df))
# with col2:
#     st.metric(label="Resolved Complaints", value=len(df[df["Status"] == "Resolved"]))
# with col3:
#     st.metric(label="Pending Complaints", value=len(df[df["Status"] == "Pending"]))
# with col4:
#     st.metric(label="Escalated Complaints", value=len(df[df["Status"] == "Escalated"]))

# # Charts Section
# st.subheader("Visualization of Complaint Trends")

# # Barchart
# fig_bar = px.bar(filtered_df, x="Category", y="Resolution_Time", color="Status", title="Resolution Time by Category")
# st.plotly_chart(fig_bar)

# # Line Chart with Animation
# st.subheader("Complaint Trends Over Time (Animation)")
# fig_line = px.line(
#     filtered_df,
#     x="Complaint_Date",
#     y="Resolution_Time",
#     color="Category",
#     animation_frame=filtered_df["Complaint_Date"].dt.strftime("%Y-%m-%d"),
#     title="Daily Resolution Time",
# )
# st.plotly_chart(fig_line)

# # Donut Chart
# st.subheader("Complaint Status Distribution")
# fig_donut = px.pie(
#     filtered_df,
#     names="Status",
#     values="Resolution_Time",
#     hole=0.4,
#     title="Proportion of Complaint Status"
# )
# st.plotly_chart(fig_donut)

# # Camera Input for Escalation
# photo = st.camera_input("Capture Photo for Escalation")
# if photo:
#     st.image(photo, caption="Captured Escalation Photo")

# # Spinner for Complaint Resolution Simulation
# if st.button("Simulate Resolution"):
#     with st.spinner("Resolving complaints..."):
#         time.sleep(5)
#     st.success("All complaints resolved successfully!")

# # Validation
# if filtered_df.empty:
#     st.warning("No complaints match the current filters!")
# else:
#     st.dataframe(filtered_df)










# import streamlit as st
# import mysql.connector
# from mysql.connector import Error

# # --- Database Connection ---
# try:
#     connection = mysql.connector.connect(
#         host="localhost",       # Replace with your MySQL host
#         user="root",            # Replace with your MySQL username
#         password="Avaya@123abc",    # Replace with your MySQL password
#         database="company_db"   # Replace with your database name
#     )
#     if connection.is_connected():
#         st.success("Connected to the MySQL database!")
# except Error as e:
#     st.error(f"Error while connecting to MySQL: {e}")
#     connection = None

# # --- Streamlit App ---
# st.title("Employee Management System")

# # --- Sidebar Filters ---
# st.sidebar.header("Filters")

# # Sidebar filter for Employee ID (shared by Update and Delete sections)
# emp_id = st.sidebar.number_input("Enter Employee ID", min_value=1, key="filter_emp_id")

# # Sidebar filter for New Salary (only for Update Salary section)
# new_salary = st.sidebar.number_input("Enter New Salary", min_value=0, step=1000, key="filter_new_salary")

# # --- Update Employee Salary ---
# st.subheader("Update Employee Salary")
# if st.button("Update Salary"):
#     if connection:
#         try:
#             cursor = connection.cursor()
#             # Check if employee exists
#             cursor.execute("SELECT * FROM employees WHERE id = %s", (emp_id,))
#             employee = cursor.fetchone()
#             if employee:
#                 # Update salary
#                 query = "UPDATE employees SET salary = %s WHERE id = %s"
#                 cursor.execute(query, (new_salary, emp_id))
#                 connection.commit()
#                 st.success(f"Salary updated to {new_salary} for Employee ID {emp_id}.")
#             else:
#                 st.warning(f"No employee found with ID {emp_id}.")
#         except Error as e:
#             st.error(f"Error while updating data: {e}")

# # --- Delete Employee Record ---
# st.subheader("Delete Employee Record")
# if st.button("Delete Employee"):
#     if connection:
#         try:
#             cursor = connection.cursor()
#             # Check if employee exists
#             cursor.execute("SELECT * FROM employees WHERE id = %s", (emp_id,))
#             employee = cursor.fetchone()
#             if employee:
#                 # Delete record
#                 query = "DELETE FROM employees WHERE id = %s"
#                 cursor.execute(query, (emp_id,))
#                 connection.commit()
#                 st.success(f"Employee record with ID {emp_id} deleted successfully.")
#             else:
#                 st.warning(f"No employee found with ID {emp_id}.")
#         except Error as e:
#             st.error(f"Error while deleting data: {e}")

# # --- Show All Employees with Expander ---
# with st.expander("Show All Employees"):
#     if connection:
#         try:
#             cursor = connection.cursor()
#             query = "SELECT * FROM employees"
#             cursor.execute(query)
#             records = cursor.fetchall()
#             if records:
#                 st.write("Employee Data:")
#                 for row in records:
#                     st.write(f"ID: {row[0]}, Name: {row[1]}, Email: {row[2]}, Department: {row[3]}, Salary: {row[4]}")
#             else:
#                 st.warning("No employee records found.")
#         except Error as e:
#             st.error(f"Error retrieving data: {e}")

# # --- Close Connection ---
# if st.button("Close Connection"):
#     if connection and connection.is_connected():
#         connection.close()
#         st.warning("Database connection closed.")












# import streamlit as st
# import mysql.connector
# from mysql.connector import Error

# # --- Database Connection ---
# try:
#     connection = mysql.connector.connect(
#         host="localhost",       # Replace with your MySQL host
#         user="root",            # Replace with your MySQL username
#         password="Avaya@123abc",    # Replace with your MySQL password
#         database="company_db"   # Replace with your database name
#     )
#     if connection.is_connected():
#         st.success("Connected to the MySQL database!")
# except Error as e:
#     st.error(f"Error while connecting to MySQL: {e}")
#     connection = None

# # --- Streamlit App ---
# st.title("Employee Management System")

# # --- Sidebar Filters ---
# st.sidebar.header("Filters")

# # Sidebar filter for Employee ID (shared by Update and Delete sections)
# emp_id = st.sidebar.number_input("Enter Employee ID", min_value=1, key="filter_emp_id")

# # Sidebar filter for New Salary (only for Update Salary section)
# new_salary = st.sidebar.number_input("Enter New Salary", min_value=0, step=1000, key="filter_new_salary")

# # --- Update Employee Salary ---
# st.subheader("Update Employee Salary")
# if st.button("Update Salary"):
#     if connection:
#         try:
#             cursor = connection.cursor()
#             # Check if employee exists
#             cursor.execute("SELECT * FROM employees WHERE id = %s", (emp_id,))
#             employee = cursor.fetchone()
#             if employee:
#                 # Update salary
#                 query = "UPDATE employees SET salary = %s WHERE id = %s"
#                 cursor.execute(query, (new_salary, emp_id))
#                 connection.commit()
#                 st.success(f"Salary updated to {new_salary} for Employee ID {emp_id}.")
#             else:
#                 st.warning(f"No employee found with ID {emp_id}.")
#         except Error as e:
#             st.error(f"Error while updating data: {e}")

# # --- Delete Employee Record ---
# st.subheader("Delete Employee Record")
# if st.button("Delete Employee"):
#     if connection:
#         try:
#             cursor = connection.cursor()
#             # Check if employee exists
#             cursor.execute("SELECT * FROM employees WHERE id = %s", (emp_id,))
#             employee = cursor.fetchone()
#             if employee:
#                 # Delete record
#                 query = "DELETE FROM employees WHERE id = %s"
#                 cursor.execute(query, (emp_id,))
#                 connection.commit()
#                 st.success(f"Employee record with ID {emp_id} deleted successfully.")
#             else:
#                 st.warning(f"No employee found with ID {emp_id}.")
#         except Error as e:
#             st.error(f"Error while deleting data: {e}")

# # --- Show All Employees ---
# if st.button("Show All Employees"):
#     if connection:
#         try:
#             cursor = connection.cursor()
#             query = "SELECT * FROM employees"
#             cursor.execute(query)
#             records = cursor.fetchall()
#             if records:
#                 st.write("Employee Data:")
#                 for row in records:
#                     st.write(f"ID: {row[0]}, Name: {row[1]}, Email: {row[2]}, Department: {row[3]}, Salary: {row[4]}")
#             else:
#                 st.warning("No employee records found.")
#         except Error as e:
#             st.error(f"Error retrieving data: {e}")

# # --- Close Connection ---
# if st.button("Close Connection"):
#     if connection and connection.is_connected():
#         connection.close()
#         st.warning("Database connection closed.")










# import streamlit as st
# import mysql.connector
# from mysql.connector import Error

# # --- Database Connection ---
# try:
#     connection = mysql.connector.connect(
#         host="localhost",       # Replace with your MySQL host
#         user="root",            # Replace with your MySQL username
#         password="Avaya@123abc",    # Replace with your MySQL password
#         database="company_db"   # Replace with your database name
#     )
#     if connection.is_connected():
#         st.success("Connected to the MySQL database!")
# except Error as e:
#     st.error(f"Error while connecting to MySQL: {e}")
#     connection = None

# # --- Streamlit App ---
# st.title("Employee Management System")

# # --- Update Employee Salary ---
# # --- Update Employee Salary ---
# st.subheader("Update Employee Salary")
# emp_id = st.number_input("Enter Employee ID", min_value=1, key="update_id")
# new_salary = st.number_input("Enter New Salary", min_value=0, step=1000, key="update_salary")

# if st.button("Update Salary"):
#     if connection:
#         try:
#             cursor = connection.cursor()
#             # Check if employee exists
#             cursor.execute("SELECT * FROM employees WHERE id = %s", (emp_id,))
#             employee = cursor.fetchone()
#             if employee:
#                 # Update salary
#                 query = "UPDATE employees SET salary = %s WHERE id = %s"
#                 cursor.execute(query, (new_salary, emp_id))
#                 connection.commit()
#                 st.success(f"Salary updated to {new_salary} for Employee ID {emp_id}.")
#             else:
#                 st.warning(f"No employee found with ID {emp_id}.")
#         except Error as e:
#             st.error(f"Error while updating data: {e}")

# # --- Delete Employee Record ---
# st.subheader("Delete Employee Record")
# delete_id = st.number_input("Enter Employee ID to Delete", min_value=1, key="delete_id")

# if st.button("Delete Employee"):
#     if connection:
#         try:
#             cursor = connection.cursor()
#             # Check if employee exists
#             cursor.execute("SELECT * FROM employees WHERE id = %s", (delete_id,))
#             employee = cursor.fetchone()
#             if employee:
#                 # Delete record
#                 query = "DELETE FROM employees WHERE id = %s"
#                 cursor.execute(query, (delete_id,))
#                 connection.commit()
#                 st.success(f"Employee record with ID {delete_id} deleted successfully.")
#             else:
#                 st.warning(f"No employee found with ID {delete_id}.")
#         except Error as e:
#             st.error(f"Error while deleting data: {e}")


# # --- Show All Employees ---
# if st.button("Show All Employees"):
#     if connection:
#         try:
#             cursor = connection.cursor()
#             query = "SELECT * FROM employees"
#             cursor.execute(query)
#             records = cursor.fetchall()
#             if records:
#                 st.write("Employee Data:")
#                 for row in records:
#                     st.write(f"ID: {row[0]}, Name: {row[1]}, Email: {row[2]}, Department: {row[3]}, Salary: {row[4]}")
#             else:
#                 st.warning("No employee records found.")
#         except Error as e:
#             st.error(f"Error retrieving data: {e}")

# # --- Close Connection ---
# if st.button("Close Connection"):
#     if connection and connection.is_connected():
#         connection.close()
#         st.warning("Database connection closed.")











# import streamlit as st
# import pandas as pd

# # Mock GDP data for countries (in billions) from 1960 to 2022
# data = {
#     'Country': ['AFE', 'AFW', 'AGO', 'AND', 'ARG'],
#     '1960': [1185, 875, 107, 3, 631],
#     '1965': [1200, 890, 110, 4, 640],
#     '1970': [1250, 900, 115, 5, 650],
#     '1975': [1300, 920, 120, 6, 660],
#     '1980': [1350, 950, 130, 7, 670],
#     '1985': [1400, 980, 140, 8, 680],
#     '1990': [1450, 1010, 150, 9, 690],
#     '1995': [1500, 1040, 160, 10, 700],
#     '2000': [1550, 1070, 170, 11, 710],
#     '2005': [1600, 1100, 180, 12, 720],
#     '2010': [1700, 1150, 200, 13, 730],
#     '2015': [1800, 1200, 210, 14, 740],
#     '2020': [1900, 1250, 220, 15, 750],
#     '2022': [1950, 1300, 230, 16, 760],
# }

# # Convert the data into a DataFrame
# df = pd.DataFrame(data)

# # Set the layout for the page
# st.set_page_config(page_title="GDP Dashboard", layout="wide")

# # Sidebar Section for user inputs
# st.sidebar.title("GDP Dashboard Settings")
# st.sidebar.text("Select year and countries for GDP view")

# # User input for year selection in sidebar
# years = [str(year) for year in range(1960, 2023)]  # List of years from 1960 to 2022
# selected_year = st.sidebar.slider("Select a Year", min_value=1960, max_value=2022, value=2022, step=1)

# # User input for country selection in sidebar
# countries = df['Country'].tolist()
# selected_countries = st.sidebar.multiselect("Select Countries to View GDP Data", countries)

# # Main Header Section
# st.title("🌎 GDP Dashboard")
# st.text("Explore GDP data from the World Bank Open Data (mock dataset)")

# # Convert selected_year to string to match column names
# selected_year_str = str(selected_year)

# # Filter data based on selected countries and year
# filtered_data = df[df['Country'].isin(selected_countries)]

# # Display GDP for selected year
# if st.button("Show GDP for Selected Year"):
#     st.subheader(f"GDP for {selected_year}")
#     for country in selected_countries:
#         country_data = filtered_data[filtered_data['Country'] == country]
#         # Access GDP using the selected year (now as string)
#         gdp_value = country_data[selected_year_str].values[0]
#         st.write(f"{country} GDP in {selected_year}: {gdp_value} Billion USD")

# # Display GDP over time for selected countries
# if st.button("Show GDP Over Time"):
#     st.subheader("GDP Over Time")
#     for country in selected_countries:
#         country_data = filtered_data[filtered_data['Country'] == country]
#         country_data = country_data.set_index('Country').T  # Transpose to have years as rows
#         st.line_chart(country_data)







# import pandas as pd
# import streamlit as st

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

#     # Display the dataset
#     st.write(mydata)

#     # Count Leave Taken distribution
#     countleave = mydata['Leave_Taken'].value_counts().reset_index()
#     countleave.columns = ['Leave_Taken', 'Count']

#     # Display bar chart
#     st.bar_chart(countleave.set_index('Leave_Taken'))






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

















# SELF-EXCLUSION


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








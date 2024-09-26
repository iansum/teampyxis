import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import networkx as nx
import matplotlib.pyplot as plt

# Initialize session state to hold the uploaded data across views
if 'data' not in st.session_state:
    st.session_state['data'] = None

# # Title of the app
# st.title('Mapping of Accident Prone Zone for Data Visualization and Analysis')

# Sidebar with options
with st.sidebar:
    st.write("Navigation")
    view_option = st.radio(
        "Choose a View:",
        ('Upload and Map Data', 'Apply kmeans', 'Apply apriori'),
        index=0
    )

# Helper function to generate dynamic tooltip
def generate_tooltip(row):
    return (
        f"Location: {row.get('location', 'Unknown')}<br>"
        f"Offense: {row.get('offense', 'Unknown')}<br>"
        f"Date: {row.get('date', 'Unknown')}<br>"
        f"Cause: {row.get('cause of accidents', 'Unknown')}<br>"
        f"Victim's Age: {row.get('victims age', 'Unknown')}<br>"
        f"Suspect's Age: {row.get('suspects age', 'Unknown')}"
    )

# Define a function to compute the silhouette score for different k values
# that will return optimal_K, optimal silhouette score
def find_optimal_k(data, max_k=20):
    silhouette_scores = []
    k_range = range(2, max_k + 1)
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(data)
        score = silhouette_score(data, labels)
        silhouette_scores.append(score)
    
    # Find the k with the highest silhouette score
    optimal_k = k_range[np.argmax(silhouette_scores)]
    optimal_score = max(silhouette_scores)
    
    return optimal_k, optimal_score



# Logic for each tab view
if view_option == 'Upload and Map Data':
    st.title("Upload CSV and Visualize Accident Data")

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    center_lat, center_lon = 13.4246, 123.3904
    zoom_level = 12
    m = folium.Map(location=[center_lat, center_lon], zoom_start=zoom_level)

    bounds = [[13.3, 123.3], [13.6, 123.5]]
    m.fit_bounds(bounds)

    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        data.columns = map(str.lower, data.columns)
        st.session_state['data'] = data

        st.write("Uploaded Data:")
        st.write(data.head())

        if 'latitude' in data.columns and 'longitude' in data.columns:
            if 'location' in data.columns:
                location_frequency = data['location'].value_counts().reset_index()
                location_frequency.columns = ['Location', 'Frequency']
                st.write("Frequency of Accidents by Location:")
                st.write(location_frequency)

            center_lat = data['latitude'].mean()
            center_lon = data['longitude'].mean()
            m = folium.Map(location=[center_lat, center_lon], zoom_start=13)

            for index, row in data.iterrows():
                folium.CircleMarker(
                    location=[row['latitude'], row['longitude']],
                    radius=4,
                    color='blue',
                    fill=True,
                    fill_color='blue',
                    tooltip=generate_tooltip(row)
                ).add_to(m)
        else:
            st.error("The uploaded file must contain 'Latitude' and 'Longitude' columns.")

    else:
        st.write("Please upload a CSV file to visualize the map.")

    st_folium(m, width=700, height=500)

elif view_option == 'Apply kmeans':
    st.title("Apply K-means clustering")

    if st.session_state['data'] is None:
        st.error("Please upload data in the 'Upload and Map Data' view first.")
    else:
        data = st.session_state['data']


        # --- K-means clustering: Latitude vs Longitude
        if 'latitude' in data.columns and 'longitude' in data.columns:
            st.header("Accident location clusters using latitude and longitude")


            # Input for number of clusters
            num_clusters = st.number_input('Select Number of Clusters:', min_value=2, max_value=10, value=3)

            # Apply K-means clustering based on user's selected number of clusters
            coords = data[['latitude', 'longitude']]
            kmeans = KMeans(n_clusters=num_clusters, random_state=42)
            data['cluster'] = kmeans.fit_predict(coords)

            # Calculate silhouette score
            silhouette_avg = silhouette_score(coords, data['cluster'])
            st.write(f"Silhouette Score for {num_clusters} clusters: {silhouette_avg:.3f}")

            st.write("Clustering Applied. Cluster Assignments:")
            st.write(data[['latitude', 'longitude', 'cluster']].head())

            # Create a Folium map to visualize the clusters
            m = folium.Map(location=[coords['latitude'].mean(), coords['longitude'].mean()], zoom_start=13)

            # Define colors for each cluster
            cluster_colors = ['red', 'blue', 'green', 'purple', 'orange', 'darkred', 'lightred', 'beige', 'darkblue', 'lightblue']

            # Add markers to the map for each location, with tooltips for additional information
            for index, row in data.iterrows():
                folium.CircleMarker(
                    location=[row['latitude'], row['longitude']],
                    radius=6,
                    color=cluster_colors[row['cluster'] % len(cluster_colors)],
                    fill=True,
                    fill_color=cluster_colors[row['cluster'] % len(cluster_colors)],
                    tooltip=generate_tooltip(row)  # Using the dynamic tooltip function
                ).add_to(m)

            # Display the map with clusters
            st_folium(m, width=700, height=500)

            # Additional section: Automatically calculate and display the optimal number of clusters based on silhouette scores
            st.write("Automatically calculating the optimal number of clusters based on silhouette scores:")

            min_clusters = 2
            max_clusters = 10
            silhouette_scores = {}

            # Loop through possible cluster numbers and calculate silhouette scores
            for k in range(min_clusters, max_clusters + 1):
                kmeans = KMeans(n_clusters=k, random_state=42)
                labels = kmeans.fit_predict(coords)
                score = silhouette_score(coords, labels)
                silhouette_scores[k] = score

            # Find the optimal number of clusters
            optimal_k = max(silhouette_scores, key=silhouette_scores.get)
            optimal_silhouette_score = silhouette_scores[optimal_k]

            st.write(f"The optimal number of clusters is {optimal_k} with a Silhouette Score of {optimal_silhouette_score:.3f}")

            # Automatically apply K-means clustering with the optimal number of clusters
            kmeans_opt = KMeans(n_clusters=optimal_k, random_state=42)
            data['cluster'] = kmeans_opt.fit_predict(coords)

            st.write(f"K-means clustering applied with {optimal_k} clusters.")
            st.write(data[['latitude', 'longitude', 'cluster']].head())

            # Create a Folium map for the optimal clustering
            m_opt = folium.Map(location=[coords['latitude'].mean(), coords['longitude'].mean()], zoom_start=13)

            # Add markers for the optimal clustering
            for index, row in data.iterrows():
                folium.CircleMarker(
                    location=[row['latitude'], row['longitude']],
                    radius=6,
                    color=cluster_colors[row['cluster'] % len(cluster_colors)],
                    fill=True,
                    fill_color=cluster_colors[row['cluster'] % len(cluster_colors)],
                    tooltip=generate_tooltip(row)
                ).add_to(m_opt)

            # Display the map with optimal clusters
            st_folium(m_opt, width=700, height=500)

        else:
            st.error("The uploaded data must contain 'Latitude' and 'Longitude' columns for clustering.")



        # --- K-means clustering: Victim's Age and Suspect's Age
        if 'victims age' in data.columns and 'suspects age' in data.columns:
            # Extract the relevant columns 'Victims Age' and 'Suspects Age' for clustering
            # Converting the columns to numeric, as they might be strings
            data['victims age'] = pd.to_numeric(data['victims age'], errors='coerce')
            data['suspects age'] = pd.to_numeric(data['suspects age'], errors='coerce')

            # Drop rows with missing values in the 'Victims Age' and 'Suspects Age' columns
            age_data = data[['victims age', 'suspects age']].dropna()

            st.header("K-Means Clustering for Victims vs Suspects Age")

            # Find the optimal k using silhouette method
            optimal_k, optimal_silhouette_score = find_optimal_k(age_data)

            # Display the optimal k and silhouette score
            st.write(f"**Optimal number of clusters (k):** {optimal_k}")
            st.write(f"**Optimal silhouette score:** {optimal_silhouette_score:.3f}") 


            # Now perform k-means clustering with the optimal_k
            kmeans = KMeans(n_clusters=optimal_k, random_state=42)
            age_data['Cluster'] = kmeans.fit_predict(age_data)

            # Plotting
            fig, ax = plt.subplots(figsize=(8, 6))
            scatter = ax.scatter(age_data['victims age'], age_data['suspects age'], c=age_data['Cluster'], cmap='viridis')
            ax.set_xlabel('Victims Age')
            ax.set_ylabel('Suspects Age')
            ax.set_title(f'K-Means Clustering for Victims vs Suspects Age with k={optimal_k}')
            plt.colorbar(scatter, label='Cluster')

            # Display plot in Streamlit
            st.pyplot(fig)
 

        # --- K-Means Clustering for Time of Day vs Cause of Accident

        if 'time' in data.columns and 'cause of accidents' in data.columns:
            st.header("K-Means Clustering for Time of Day vs Cause of Accident")

            # Map "Time of Day" to numerical values starting from 0
            time_mapping = {
                'Morning': 0,
                'Afternoon': 1,
                'Evening': 2,
                'Night': 3
            }
            data['time_numeric'] = data['time'].map(time_mapping)

            # Check if mapping was successful
            if data['time_numeric'].isnull().any():
                st.error("There are unmapped 'time of day' values. Please make sure all time periods are defined.")

            # Encoding "Cause of Accident" column into numeric form
            cause_of_accident_encoded = pd.factorize(data['cause of accidents'])[0]

            # Drop any rows with missing values in 'time_numeric' or 'cause of accidents'
            time_accident_data = data[['time_numeric', 'cause of accidents']].dropna()
            time_accident_data['cause of accidents'] = cause_of_accident_encoded

            # Find the optimal k using silhouette method
            optimal_k, optimal_silhouette_score = find_optimal_k(time_accident_data[['time_numeric', 'cause of accidents']])

            # Display the optimal k and silhouette score
            st.write(f"**Optimal number of clusters (k):** {optimal_k}")
            st.write(f"**Optimal silhouette score:** {optimal_silhouette_score:.3f}")

            # Now perform k-means clustering with the optimal_k
            kmeans = KMeans(n_clusters=optimal_k, random_state=42)
            time_accident_data['Cluster'] = kmeans.fit_predict(time_accident_data[['time_numeric', 'cause of accidents']])

            # Plotting the clustering results
            fig, ax = plt.subplots(figsize=(8, 6))
            scatter = ax.scatter(time_accident_data['time_numeric'], time_accident_data['cause of accidents'], c=time_accident_data['Cluster'], cmap='viridis')
            ax.set_xlabel('Time of Day (Numeric)')
            ax.set_ylabel('Cause of Accident (Encoded)')
            ax.set_title(f'K-Means Clustering for Time of the day vs Cause of Accident with k={optimal_k}')
            plt.colorbar(scatter, label='Cluster')

            # Display the plot in Streamlit
            st.pyplot(fig)

            # # Optional: Show the legend for the "Time of Day" mapping
            # st.write("Time of Day Mapping:")
            # st.write(time_mapping)

        else:
            st.error("The dataset must contain 'time of day' and 'cause of accidents' columns.")


        # --- Kmeans cluster Vehicle kind vs Suspects Age
        if 'vehicle kind' in data.columns and 'suspects age' in data.columns:
            st.header("K-Means Clustering for Vehicle Kind vs Suspects Age")

            # Encoding 'Vehicle Kind' into numeric form
            vehicle_kind_encoded, vehicle_kind_categories = pd.factorize(data['vehicle kind'])
            data['vehicle_kind_numeric'] = vehicle_kind_encoded

            # Ensure 'Suspects Age' is numeric
            data['suspects age'] = pd.to_numeric(data['suspects age'], errors='coerce')

            # Drop rows with missing values in the relevant columns
            vk_age_data = data[['vehicle_kind_numeric', 'suspects age']].dropna()

            # Find the optimal k using silhouette method
            optimal_k, optimal_silhouette_score = find_optimal_k(vk_age_data)

            # Display the optimal k and silhouette score
            st.write(f"**Optimal number of clusters (k):** {optimal_k}")
            st.write(f"**Optimal silhouette score:** {optimal_silhouette_score:.3f}")

            # Now perform k-means clustering with the optimal_k
            kmeans = KMeans(n_clusters=optimal_k, random_state=42)
            vk_age_data['Cluster'] = kmeans.fit_predict(vk_age_data)

            # Plotting the clustering results
            fig, ax = plt.subplots(figsize=(8, 6))
            scatter = ax.scatter(
                vk_age_data['vehicle_kind_numeric'],
                vk_age_data['suspects age'],
                c=vk_age_data['Cluster'],
                cmap='viridis'
            )
            ax.set_xlabel('Vehicle Kind (Encoded)')
            ax.set_ylabel('Suspects Age')
            ax.set_title(f'K-Means Clustering for Vehicle Kind vs Suspects Age with k={optimal_k}')
            plt.colorbar(scatter, label='Cluster')

            # Display the plot in Streamlit
            st.pyplot(fig)

            # Optional: Show the mapping from encoded values to actual 'Vehicle Kind'
            st.write("Vehicle Kind Encoding Mapping:")
            vk_mapping = dict(enumerate(vehicle_kind_categories))
            st.write(vk_mapping)
        else:
            st.error("The dataset must contain 'vehicle kind' and 'suspects age' columns.")
            

        # --- K-means clustering: Victims Age vs Cause of Accidents
        if 'victims age' in data.columns and 'cause of accidents' in data.columns:
            st.header("K-Means Clustering for Victims Age vs Cause of Accidents")

            # Ensure 'Victims Age' is numeric
            data['victims age'] = pd.to_numeric(data['victims age'], errors='coerce')

            # Encoding 'Cause of Accidents' into numeric form
            cause_accident_encoded, cause_accident_categories = pd.factorize(data['cause of accidents'])
            data['cause_accident_numeric'] = cause_accident_encoded

            # Drop rows with missing values in the relevant columns
            va_cause_data = data[['victims age', 'cause_accident_numeric']].dropna()

            # Find the optimal k using silhouette method
            optimal_k, optimal_silhouette_score = find_optimal_k(va_cause_data)

            # Display the optimal k and silhouette score
            st.write(f"**Optimal number of clusters (k):** {optimal_k}")
            st.write(f"**Optimal silhouette score:** {optimal_silhouette_score:.3f}")

            # Now perform k-means clustering with the optimal_k
            kmeans = KMeans(n_clusters=optimal_k, random_state=42)
            va_cause_data['Cluster'] = kmeans.fit_predict(va_cause_data)

            # Plotting the clustering results
            fig, ax = plt.subplots(figsize=(8, 6))
            scatter = ax.scatter(
                va_cause_data['victims age'],
                va_cause_data['cause_accident_numeric'],
                c=va_cause_data['Cluster'],
                cmap='viridis'
            )
            ax.set_xlabel('Victims Age')
            ax.set_ylabel('Cause of Accident (Encoded)')
            ax.set_title(f'K-Means Clustering for Victims Age vs Cause of Accidents with k={optimal_k}')
            plt.colorbar(scatter, label='Cluster')

            # Display the plot in Streamlit
            st.pyplot(fig)

            # Optional: Show the mapping from encoded values to actual 'Cause of Accidents'
            st.write("Cause of Accidents Encoding Mapping:")
            cause_mapping = dict(enumerate(cause_accident_categories))
            st.write(cause_mapping)
        else:
            st.error("The dataset must contain 'victims age' and 'cause of accidents' columns.")


        # --- K-means clustering: Vehicle Kind vs Victims Age
        if 'vehicle kind' in data.columns and 'victims age' in data.columns:
            st.header("K-Means Clustering for Vehicle Kind vs Victims Age")

            # Encoding 'Vehicle Kind' into numeric form
            vehicle_kind_encoded, vehicle_kind_categories = pd.factorize(data['vehicle kind'])
            data['vehicle_kind_numeric'] = vehicle_kind_encoded

            # Ensure 'Victims Age' is numeric
            data['victims age'] = pd.to_numeric(data['victims age'], errors='coerce')

            # Drop rows with missing values in the relevant columns
            vk_victim_age_data = data[['vehicle_kind_numeric', 'victims age']].dropna()

            # Find the optimal k using silhouette method
            optimal_k, optimal_silhouette_score = find_optimal_k(vk_victim_age_data)

            # Display the optimal k and silhouette score
            st.write(f"**Optimal number of clusters (k):** {optimal_k}")
            st.write(f"**Optimal silhouette score:** {optimal_silhouette_score:.3f}")

            # Now perform k-means clustering with the optimal_k
            kmeans = KMeans(n_clusters=optimal_k, random_state=42)
            vk_victim_age_data['Cluster'] = kmeans.fit_predict(vk_victim_age_data)

            # Plotting the clustering results
            fig, ax = plt.subplots(figsize=(8, 6))
            scatter = ax.scatter(
                vk_victim_age_data['vehicle_kind_numeric'],
                vk_victim_age_data['victims age'],
                c=vk_victim_age_data['Cluster'],
                cmap='viridis'
            )
            ax.set_xlabel('Vehicle Kind (Encoded)')
            ax.set_ylabel('Victims Age')
            ax.set_title(f'K-Means Clustering for Vehicle Kind vs Victims Age with k={optimal_k}')
            plt.colorbar(scatter, label='Cluster')

            # Display the plot in Streamlit
            st.pyplot(fig)

            # Optional: Show the mapping from encoded values to actual 'Vehicle Kind'
            st.write("Vehicle Kind Encoding Mapping:")
            vk_mapping = dict(enumerate(vehicle_kind_categories))
            st.write(vk_mapping)
        else:
            st.error("The dataset must contain 'vehicle kind' and 'victims age' columns.")


        # --- K-means clustering: Year vs Cause of Accidents
        if ('year' in data.columns or 'date' in data.columns) and 'cause of accidents' in data.columns:
            st.header("K-Means Clustering for Year vs Cause of Accidents")

            # If 'year' is not in columns, try to extract 'year' from 'date' column
            if 'year' not in data.columns and 'date' in data.columns:
                # Convert 'date' to datetime and extract year
                data['date'] = pd.to_datetime(data['date'], errors='coerce')
                data['year'] = data['date'].dt.year

            # Ensure 'year' is numeric
            data['year'] = pd.to_numeric(data['year'], errors='coerce')

            # Encoding 'Cause of Accidents' into numeric form
            cause_accident_encoded, cause_accident_categories = pd.factorize(data['cause of accidents'])
            data['cause_accident_numeric'] = cause_accident_encoded

            # Drop rows with missing values in the relevant columns
            year_cause_data = data[['year', 'cause_accident_numeric']].dropna()

            if year_cause_data.empty:
                st.error("No data available after removing rows with missing values.")
            else:
                # Find the optimal k using silhouette method
                optimal_k, optimal_silhouette_score = find_optimal_k(year_cause_data)

                # Display the optimal k and silhouette score
                st.write(f"**Optimal number of clusters (k):** {optimal_k}")
                st.write(f"**Optimal silhouette score:** {optimal_silhouette_score:.3f}")

                # Now perform k-means clustering with the optimal_k
                kmeans = KMeans(n_clusters=optimal_k, random_state=42)
                year_cause_data['Cluster'] = kmeans.fit_predict(year_cause_data)

                # Plotting the clustering results
                fig, ax = plt.subplots(figsize=(8, 6))
                scatter = ax.scatter(
                    year_cause_data['year'],
                    year_cause_data['cause_accident_numeric'],
                    c=year_cause_data['Cluster'],
                    cmap='viridis'
                )
                ax.set_xlabel('Year')
                ax.set_ylabel('Cause of Accident (Encoded)')
                ax.set_title(f'K-Means Clustering for Year vs Cause of Accidents with k={optimal_k}')
                plt.colorbar(scatter, label='Cluster')

                # Display the plot in Streamlit
                st.pyplot(fig)

                # Optional: Show the mapping from encoded values to actual 'Cause of Accidents'
                st.write("Cause of Accidents Encoding Mapping:")
                cause_mapping = dict(enumerate(cause_accident_categories))
                st.write(cause_mapping)
        else:
            st.error("The dataset must contain 'year' (or 'date') and 'cause of accidents' columns.")

elif view_option == 'Apply apriori':
    st.title("Apriori Algorithm Analysis")

    if st.session_state['data'] is None:
        st.error("Please upload data in the 'Upload and Map Data' view first.")
    else:
        data = st.session_state['data']

        # Select columns for Apriori analysis
        st.write("Select the columns you want to include in the Apriori analysis:")
        # Get a list of columns that are categorical or can be treated as such
        potential_columns = data.select_dtypes(include=['object', 'category']).columns.tolist()
        selected_columns = st.multiselect("Columns", potential_columns, default=potential_columns)

        if not selected_columns:
            st.error("Please select at least one column.")
        else:
            # Preprocess the data for Apriori
            st.write("Preparing data for Apriori analysis...")
            # Convert selected columns to string and handle missing values
            apriori_data = data[selected_columns].astype(str).fillna('Missing')

            # Create a list of transactions
            transactions = apriori_data.values.tolist()

            # Apply the TransactionEncoder
            te = TransactionEncoder()
            te_ary = te.fit(transactions).transform(transactions)
            df_te = pd.DataFrame(te_ary, columns=te.columns_)

            # Get minimum support from user
            min_support = st.slider("Select minimum support:", min_value=0.01, max_value=1.0, value=0.1, step=0.01)

            # Apply Apriori algorithm
            frequent_itemsets = apriori(df_te, min_support=min_support, use_colnames=True)

            if frequent_itemsets.empty:
                st.warning("No frequent itemsets found with the selected minimum support.")
            else:
                st.write("Frequent Itemsets:")
                itemsets = frequent_itemsets['itemsets'].apply(lambda x: ', '.join(list(x)) if isinstance(x, frozenset) else x)
                frequent_combined = pd.concat([frequent_itemsets['support'], itemsets], axis=1)
                st.write(frequent_combined)

                # Get minimum confidence from user
                min_confidence = st.slider("Select minimum confidence:", min_value=0.01, max_value=1.0, value=0.5, step=0.01)

                # Generate association rules
                rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)

                if rules.empty:
                    st.warning("No association rules found with the selected minimum confidence.")
                else:
                    # Convert frozensets in 'antecedents' and 'consequents' to lists or strings
                    antecedent = rules['antecedents'].apply(lambda x: ', '.join(list(x)) if isinstance(x, frozenset) else x)
                    consequent = rules['consequents'].apply(lambda x: ', '.join(list(x)) if isinstance(x, frozenset) else x)


                    association_combined = pd.concat([antecedent, consequent, rules[['support', 'confidence', 'lift']]], axis=1)


                    # Display the modified association rules
                    st.write("Association Rules:")
                    st.write(association_combined)

                    # Optional: Visualize the rules using a network graph
                    st.write("Association Rules Network Graph:")
                    G = nx.DiGraph()

                    # Add nodes and edges
                    for _, row in rules.iterrows():
                        for antecedent in row['antecedents']:
                            for consequent in row['consequents']:
                                G.add_edge(antecedent, consequent, weight=row['confidence'])

                    # Draw the network graph
                    plt.figure(figsize=(12, 8))
                    pos = nx.spring_layout(G, k=0.5)
                    nx.draw_networkx_nodes(G, pos, node_size=700)
                    nx.draw_networkx_edges(G, pos, arrowstyle='->', arrowsize=20)
                    nx.draw_networkx_labels(G, pos, font_size=10)
                    plt.axis('off')
                    st.pyplot(plt)
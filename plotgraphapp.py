import streamlit as st
import pandas as pd
import plotly.express as px
import os
import folium
from streamlit_folium import st_folium
from streamlit_js_eval import streamlit_js_eval
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import joblib
from pathlib import Path

# --- 1. SETUP & PATHS ---
st.set_page_config(page_title="Coastal Simulation Analysis", page_icon="🌊", layout="wide")

base_dir = Path(__file__).parent

scenario_map = {
    'Without Sand Engine and Seagrass':   os.path.join(base_dir, "Without_Sandengine_and_Seagrass"),
    'With Sand Engine':                   os.path.join(base_dir, "With_Sandengine"),
    'With Seagrass':                      os.path.join(base_dir, "With_Seagrass"),
    'With Sand Engine and Seagrass':      os.path.join(base_dir, "With_Sandengine_and_Seagrass")
}

wave_folders = ['0.5', '1', '2', '3']
stations = [1, 2, 3, 4, 5, 6, 7]

param_map = {
    'Water Depth': ['water depth', 'depth'],
    'Significant wave height': ['significant wave height', 'wave height', 'hs'],
    'Depth average velocity': ['velocity', 'depth averaged velocity'],
    'Depth average sediment transport (d. a. sediment transport)': ['transport', 'suspended transport']
}

coord_metadata = {
    1: {"x": "TBC", "y": "TBC"}, 2: {"x": "TBC", "y": "TBC"},
    3: {"x": "TBC", "y": "TBC"}, 4: {"x": "TBC", "y": "TBC"},
    5: {"x": "TBC", "y": "TBC"}, 6: {"x": "TBC", "y": "TBC"},
    7: {"x": 502801.0, "y": 422013.0} 
}

# --- 2. HELPERS ---
def find_file_robustly(folder_path, station_num, param_keywords):
    if not os.path.exists(folder_path): return None
    for f in os.listdir(folder_path):
        f_lower = f.lower()
        station_patterns = [f"river {station_num}", f"river{station_num}", f"r{station_num}"]
        has_station = any(p in f_lower for p in station_patterns)
        has_param = any(kw in f_lower for kw in param_keywords)
        if has_station and has_param and f.endswith('.csv'):
            return os.path.join(folder_path, f)
    return None

@st.cache_data
def convert_df_to_csv(df):
    return df.to_csv(index=False).encode('utf-8')

# --- 3. CLEAN SIDEBAR CONTROLS & README ---
if "show_controls" not in st.session_state:  # initialise show control session state (variable)
    st.session_state.show_controls = False



if not st.session_state.show_controls:

    col1,  col2 = st.sidebar.columns([1, 1])
    with col1:
        st.image(os.path.join(base_dir, "dashboard_files", 'logo_MMU.png'),width=200 ,link='https://www.mmu.ac.uk/')
    with col2:
        st.image(os.path.join(base_dir, 'logo_Liverpool.svg'),width=200 ,link='https://www.liverpool.ac.uk/')

    st.markdown("""
        <style>
            /* Remove gap below image */
            [data-testid="stSidebar"] [data-testid="stImage"] {
                margin-bottom: 0rem !important;
            }

            /* Remove gap above markdown line */
            [data-testid="stSidebar"] hr {
                margin-top: -1rem !important;
            }
        </style>
    """, unsafe_allow_html=True)
    st.sidebar.markdown("---")
    st.sidebar.title("🌊 Coastal Simulation Analysis")
    st.sidebar.markdown(
        """ 
        This dashboard presents the results of **Delft 3D** hydrodynamic numerical modelling 
        and **machine learning** prediction models, simulating coastal dynamics under three 
        intervention scenarios:
        """)
    
    st.markdown(
        """
        <style>
        .pill {
                display: block;
                padding: 5px 12px;
                border-radius: 20px;
                font-size: 0.8rem;
                font-weight: 600;
                margin-bottom: 7px;
            }
        .pill-blue   { background: rgba(76,201,240,0.12);  border: 1px solid #4CC9F0; color: #4CC9F0; }
        .pill-orange { background: rgba(244,162,97,0.12);  border: 1px solid #F4A261; color: #F4A261; }
        .pill-purple { background: rgba(123,47,190,0.12);  border: 1px solid #7B2FBE; color: #7B2FBE; }
        .pill-teal   { background: rgba(42,157,143,0.15);  border: 1px solid #2A9D8F; color: #2A9D8F; }
        </style>
        """, unsafe_allow_html=True
    )
    st.sidebar.markdown(
        '<span class="pill pill-blue">🔵 Without Sand Engine and Seagrass — Baseline</span>'
        '<span class="pill pill-orange">🟠 With Sand Engine — Nourishment</span>'
        '<span class="pill pill-purple">🟣 With Seagrass — NbS</span>'
        '<span class="pill pill-teal">🟢 Sand Engine + Seagrass — NbS</span>',
        unsafe_allow_html=True,
    )
    st.sidebar.markdown(
        """
        A **sand engine** was placed near the mouth of river Leven. **Seagrass** was placed
        all over the sand engine area. The model was run across four offshore wave height conditions
        (0.5m, 1m, 2m, 3m) and measurements were recorded at 7 river stations. These river
        points are placed at an interval of 1km upstream from the sand engine.
        
        **SVR** and **GPR** models
        were trained on the simulation results to predict the significant wave height and 
        sediment transport under varying offshore wave height conditions and intervention configuration.
        """
    )

    st.sidebar.markdown("---")

    view_result_button = st.sidebar.button("📊 View Results →", use_container_width=True)
    if view_result_button:
        st.session_state.show_controls = True
        st.rerun()
    
    
    # Main page map
    st.markdown("## 🗺️ Study Area — Morecambe Bay, UK")
    st.caption("Explore the simulation boundary, sand engine location, and river monitoring stations.")

    m = folium.Map(
        location=[54.01, -3.13],
        zoom_start=10,
        tiles=folium.TileLayer("CartoDB positron",       name="CartonDB Positron")    
    )
    folium.TileLayer("Esri.WorldImagery",   name="Satellite").add_to(m)
    folium.TileLayer("CartoDB dark_matter", name="Dark").add_to(m)
    folium.TileLayer("OpenStreetMap",       name="OpenStreet Map").add_to(m)

    # Load and display GeoJSON directly
    folium.GeoJson(
        os.path.join(base_dir, "boundary.geojson"),
        name="boundary",
        tooltip=folium.Tooltip(text="Simulation Boundary"),
        popup=folium.Popup(
        html="""
                <b>Simulation Boundary</b><br>
                Location: Morecambe Bay, UK<br>
                Model: Delft3D · Period: 10 days<br>
                Wave height: 0.5, 1, 2, 3m<br>
            """,
            max_width=250,
        )
        # tooltip=folium.GeoJsonTooltip(fields=["boundary"]),  # show attribute on hover
        # popup=folium.GeoJsonPopup(fields=["name", "area"]) # show attributes on click
    ).add_to(m)
    folium.GeoJson(
        os.path.join(base_dir, "river_points.geojson"),
        name="River Points",
        marker=folium.CircleMarker(
            radius=8,
            color="white",
            weight=2,
            fill=True,
            fill_color="#1f77b4",
            fill_opacity=0.9
        ),
        tooltip=folium.GeoJsonTooltip(fields=["Name"])
    ).add_to(m)
    folium.GeoJson(
        os.path.join(base_dir,"sandengine.geojson"),
        name="Sandengine",
        tooltip=folium.Tooltip(text="Sandengine and Seagrass"),
        popup=folium.Popup(
            html="""
                    <b>Sand Engine & Seagrass</b><br>Height: 0.5 m · Radius: 1.5 km<br>
                    Scenarios: None / SE only / SE + Seagrass
        """, max_width=400
        )
    ).add_to(m)
    folium.LayerControl().add_to(m)
    st_folium(m, use_container_width=True, height=900,returned_objects=[])

if st.session_state.show_controls:
    col1,  col2 = st.sidebar.columns([1, 1])
    with col1:
        st.image(os.path.join(base_dir, 'logo_MMU.png'),width=200 ,link='https://www.mmu.ac.uk/')
    with col2:
        st.image(os.path.join(base_dir, 'logo_Liverpool.svg'),width=200 ,link='https://www.liverpool.ac.uk/')
    st.markdown("""
        <style>
            /* Remove gap below image */
            [data-testid="stSidebar"] [data-testid="stImage"] {
                margin-bottom: 0rem !important;
            }

            /* Remove gap above markdown line */
            [data-testid="stSidebar"] hr {
                margin-top: -0.5rem !important;
            }
        </style>
    """, unsafe_allow_html=True)
    st.sidebar.markdown("---")
    if st.sidebar.button("🗺️ Back to Map", use_container_width=True):
        st.session_state.show_controls = False
        st.rerun()
    st.sidebar.markdown("---")

    st.sidebar.title("⚙️ Global Controls")
    st.sidebar.markdown("### 📐 Variable")
    selected_param = st.sidebar.selectbox("Analyzed Variable:", list(param_map.keys()), label_visibility="collapsed")

    st.sidebar.markdown("### 🗂️ Scenarios")
    selected_scenarios = st.sidebar.multiselect(
        "Select models to compare:",
        options=list(scenario_map.keys()),
        default=list(scenario_map.keys())
    )
    
    st.sidebar.markdown("---")
    # --- NEW README SECTION ---
    with st.sidebar.expander("📖 How to Use"):
        st.markdown(
            "- **Variables:** Use the top dropdown to switch between Depth, Velocity, Transport, and Wave Height.\n\n"
            "- **Scenarios:** In the box above, click the 'X' next to a scenario name (e.g., 'With Sand Engine') to remove it from the graph.\n\n"
            "- **Exact Data Query:** Scroll below the graphs to extract exact numerical values for specific simulation timestamps.\n\n"
            "- **Export** — use the download button at the bottom of each tab."
        )

    # --- 4. MAIN INTERFACE ---
    st.markdown("## 🌊 Coastal Simulation Analytics")
    st.caption("Morecambe Bay · Delft3D hydrodynamic modelling · SVR & GPR surrogate models")
    st.markdown("---")
    st.markdown(
        """
            <style>
            /* ── Tabs ── */
            .stTabs [data-baseweb="tab-list"] {
                gap: 4px;
                padding: 4px 6px;
                border-radius: 10px;
                background: rgba(255,255,255,0.04);
                border: 1px solid rgba(255,255,255,0.08);
            }
            .stTabs [data-baseweb="tab"] {
                border-radius: 8px;
                padding: 6px 20px;
                font-size: 0.84rem;
                font-weight: 600;
                color: inherit;
                background: transparent;
            }
            .stTabs [aria-selected="true"] {
                background: #0077B6 !important;
                color: white !important;
            }</style>
            """,unsafe_allow_html=True
    )
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Scenario Comparison", "📍 Spatial Overview", "📈 Mean Analysis", "🔮 Prediction Models"])

    # ==========================================
    # TAB 1: SCENARIO COMPARISON (FIXED STATION)
    # ==========================================
    with tab1:
        st.subheader(f"Performance Comparison")
        st.markdown(
            f"Comparing **{selected_param}** across scenarios at a single station over time."
        )
        
        col1, col2, col3 = st.columns([1, 1, 2])
        with col1: target_wave = st.selectbox("Boundary Wave Height", wave_folders, format_func=lambda x: f"{x}m")
        with col2: target_stat = st.selectbox("Inspection Station", stations, index=6, format_func=lambda x: f"River {x}") 

        all_data = []
        if selected_scenarios:
            for label in selected_scenarios:
                folder = os.path.join(scenario_map[label], target_wave)
                target_file = find_file_robustly(folder, target_stat, param_map[selected_param])
                if target_file:
                    try:
                        df = pd.read_csv(target_file, sep=None, engine='python')
                        df['DateTime'] = pd.to_datetime(df.iloc[:, 0])
                        df['Magnitude'] = df.iloc[:, 1]
                        df['Scenario'] = label
                        all_data.append(df)
                    except: pass

            if all_data:
                plot_df = pd.concat(all_data)
                color_mapping = {'Without Sand Engine and Seagrass': '#1f77b4', 'With Sand Engine': '#ff7f0e','With Seagrass': '#7B2FBE', 'With Sand Engine and Seagrass': '#2ca02c'}
                
                fig = px.line(plot_df, x='DateTime', y='Magnitude', color='Scenario', color_discrete_map=color_mapping)
                fig.update_xaxes(range=[plot_df['DateTime'].min(), pd.to_datetime("2018-01-11 00:00:00")])
                fig.update_layout(hovermode="x unified", height=500, xaxis_title="Simulation Time (Up to Jan 11, 2018)", yaxis_title="Magnitude", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
                st.plotly_chart(fig, use_container_width=True)
                
                # --- NEW DATA QUERY BOX (TAB 1) ---
                with st.expander("🔍 Exact Data Query"):
                    st.write("Extract the exact magnitude for a specific scenario at an exact simulation timestamp.")
                    
                    # Extract unique times from the actual data so the user can only pick valid simulation times
                    available_times = plot_df['DateTime'].sort_values().unique()
                    
                    q_col1, q_col2 = st.columns(2)
                    with q_col1:
                        query_time = st.selectbox("Simulation Time:", available_times, format_func=lambda x: pd.to_datetime(x).strftime('%Y-%m-%d %H:%M'))
                    with q_col2:
                        query_scenario = st.selectbox("Scenario:", plot_df['Scenario'].unique())
                    
                    # Filter the dataframe to find the exact match
                    query_result = plot_df[(plot_df['DateTime'] == query_time) & (plot_df['Scenario'] == query_scenario)]
                    
                    st.markdown(
                        """
                        <style>
                            .metric-chip {
                                display: inline-block;
                                background: #0077B6;
                                color: white;
                                border-radius: 20px;
                                padding: 3px 14px;
                                font-weight: 700;
                                font-size: 1.05rem;
                                margin-top: 6px;
                            }</style>
                            """,unsafe_allow_html=True
                    )
                    if not query_result.empty:
                       st.markdown(
                        f"**{pd.to_datetime(query_time).strftime('%Y-%m-%d %H:%M')}** &nbsp;·&nbsp; "
                        f"**{query_scenario}**<br>"
                        f'<span class="metric-chip">{query_result["Magnitude"].values[0]:.6f}</span>',
                        unsafe_allow_html=True,
                    )
                    else:
                        st.warning("No data recorded for this exact combination.")
                    
                # --- AUTOMATED INSIGHTS GENERATOR (TAB 1) ---
                st.markdown("### Automated Scenario Analysis")
                if 'Without Sand Engine and Seagrass' in selected_scenarios and 'With Sand Engine and Seagrass' in selected_scenarios:
                    base_df = plot_df[plot_df['Scenario'] == 'Without Sand Engine and Seagrass']
                    sea_df = plot_df[plot_df['Scenario'] == 'With Sand Engine and Seagrass']
                    
                    base_mean, sea_mean = base_df['Magnitude'].mean(), sea_df['Magnitude'].mean()
                    base_max, sea_max = base_df['Magnitude'].max(), sea_df['Magnitude'].max()
                    base_std, sea_std = base_df['Magnitude'].std(), sea_df['Magnitude'].std()
                    
                    if base_mean > 0:
                        red_mean = ((base_mean - sea_mean) / base_mean) * 100
                        red_max = ((base_max - sea_max) / base_max) * 100
                        
                        st.info(f"**📉 The Obvious Trend:** The Seagrass NbS altered the *average* {selected_param.lower()} by **{red_mean:.1f}%** at River {target_stat}.")
                        
                        if red_max > red_mean:
                            st.success(f"**🌪️ Not-So-Obvious Trend (Peak Clipping):** Seagrass works harder during extremes! It reduced the *maximum peak surge* by **{red_max:.1f}%**, which is a stronger reduction than its average performance.")
                        else:
                            st.success(f"**🌪️ Not-So-Obvious Trend (Proportional Scaling):** The peak surge was reduced by **{red_max:.1f}%**. This closely tracks the average reduction, meaning the seagrass dampens the flow equally regardless of how severe the wave gets.")
                        
                        if sea_std > base_std:
                            st.warning(f"**⚠️ Non-Trend (Turbulence Anomaly):** Interestingly, while overall magnitude decreased, the *variance* (standard deviation) increased slightly with seagrass. This implies the seagrass creates localized flow disruption or turbulence, rather than a perfectly smooth decay.")
                        else:
                            st.warning(f"**⚠️ Expected Behavior (Smooth Flow):** The data shows no turbulence anomalies. The variance (standard deviation) decreased alongside the magnitude, confirming the seagrass is smoothing the flow predictably.")

                # --- EXCEL / CSV DOWNLOAD ---
                st.markdown(
                    """
                    <style>
                    .stDownloadButton > button {
                        background: #0077B6 !important;
                        color: white !important;
                        border: none !important;
                        border-radius: 8px !important;
                        font-weight: 600 !important;
                    }
                    </style>""",unsafe_allow_html=True
                )
                st.markdown("### Export Data")
                csv_data = convert_df_to_csv(plot_df)
                st.download_button(label=f"📄 Download CSV", data=csv_data, file_name=f"{selected_param.replace(' ', '_')}_River{target_stat}_{target_wave}m.csv", mime="text/csv")
                
            else:
                st.warning("No CSV data found. Ensure files are exported correctly.")

    # ==========================================
    # TAB 2: SPATIAL OVERVIEW (ALL STATIONS)
    # ==========================================
    with tab2:
        st.subheader("Spatial Attenuation")
        st.markdown(
            f"How **{selected_param}** evolves from the river mouth (Point 1) "
            "to the deepest inland station (Point 7) over time."
        )
        
        col1, col2, col3 = st.columns([1, 1, 2])
        with col1: space_wave = st.selectbox("Boundary Wave Height", wave_folders, format_func=lambda x: f"{x}m", key='space_wave')
        with col2: space_scen = st.selectbox("Simulation Scenario", list(scenario_map.keys()), key='space_scen')
        with col3:
            selected_river_points = st.multiselect("Select River Points:", ["Point 1","Point 2","Point 3","Point 4","Point 5","Point 6","Point 7"],default=["Point 1","Point 2","Point 3","Point 4","Point 5","Point 6","Point 7"],key='tab2_river_points')

        spatial_data = []
        selected_river_points_num = pd.to_numeric([r.replace("Point ", "") for r in selected_river_points])
        for s in selected_river_points_num:
            folder = os.path.join(scenario_map[space_scen], space_wave)
            target_file = find_file_robustly(folder, s, param_map[selected_param])
            if target_file:
                try:
                    df = pd.read_csv(target_file, sep=None, engine='python')
                    df['DateTime'] = pd.to_datetime(df.iloc[:, 0])
                    df['Magnitude'] = df.iloc[:, 1]
                    df['Station'] = f"Point {s}"
                    df['StationNum'] = s 
                    spatial_data.append(df)
                except: pass

        if spatial_data:
            plot_space_df = pd.concat(spatial_data)
            station_color_map = {
                'Point 1': '#1f77b4',   # blue
                'Point 2': '#ff7f0e',   # orange
                'Point 3': '#2ca02c',   # green
                'Point 4': '#d62728',   # red
                'Point 5': '#9467bd',   # purple
                'Point 6': '#8c564b',   # brown
                'Point 7': '#17becf',   # teal
            }

            fig2 = px.line(plot_space_df, x='DateTime', y='Magnitude', color='Station', color_discrete_map=station_color_map)
            fig2.update_xaxes(range=[plot_space_df['DateTime'].min(), pd.to_datetime("2018-01-11 00:00:00")])
            fig2.update_layout(hovermode="x unified", height=500, xaxis_title="Simulation Time (Up to Jan 11, 2018)", yaxis_title="Magnitude", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
            st.plotly_chart(fig2, use_container_width=True)

            # --- NEW DATA QUERY BOX (TAB 2) ---
            with st.expander("🔍 Exact Data Query"):
                st.write("Extract the exact magnitude at a specific station for an exact simulation timestamp.")
                
                available_times_space = plot_space_df['DateTime'].sort_values().unique()
                qs_col1, qs_col2 = st.columns(2)
                
                with qs_col1:
                    query_time_space = st.selectbox("Simulation Time:", available_times_space, format_func=lambda x: pd.to_datetime(x).strftime('%Y-%m-%d %H:%M'), key='qt_space')
                with qs_col2:
                    query_station = st.selectbox("Station:", plot_space_df['Station'].unique())
                    
                query_result_space = plot_space_df[(plot_space_df['DateTime'] == query_time_space) & (plot_space_df['Station'] == query_station)]
                
                if not query_result_space.empty:
                    st.markdown(
                        f"**{pd.to_datetime(query_time_space).strftime('%Y-%m-%d %H:%M')}** &nbsp;·&nbsp; "
                        f"**{query_station}**<br>"
                        f'<span class="metric-chip">{query_result_space["Magnitude"].values[0]:.6f}</span>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.warning("No data recorded for this exact combination.")

            # --- AUTOMATED INSIGHTS GENERATOR (TAB 2) ---
            st.markdown("### Spatial Flow Analysis")
            
            r1_df = plot_space_df[plot_space_df['StationNum'] == 1]
            r7_df = plot_space_df[plot_space_df['StationNum'] == 7]
            
            if not r1_df.empty and not r7_df.empty:
                r1_max = r1_df['Magnitude'].max()
                r7_max = r7_df['Magnitude'].max()
                
                absolute_max = plot_space_df['Magnitude'].max()
                max_station_df = plot_space_df[plot_space_df['Magnitude'] == absolute_max].iloc[0]
                max_station_name = max_station_df['Station']
                
                if r1_max > 0:
                    drop_pct = ((r1_max - r7_max) / r1_max) * 100
                    st.info(f"**📉 The Obvious Trend (Inland Decay):** As expected, energy dissipates as it moves inland. The peak {selected_param.lower()} dropped by **{drop_pct:.1f}%** from the river mouth (River 1) to the deepest inland point (River 7).")
                
                if max_station_name != "River 1":
                    st.warning(f"**🌊 Not-So-Obvious Trend (Shoaling Effect):** Counter-intuitively, the highest energy was *not* at the river mouth! The maximum occurred at **{max_station_name}**. This non-linear trend suggests a 'shoaling' effect or local bottleneck.")
                else:
                    st.success(f"**🌊 Expected Trend (No Shoaling):** The highest energy occurred directly at River 1. The model did not detect any weird bottlenecks or wave-stacking (shoaling) further down the channel.")
                    
                st.info("**🔍 Non-Trend Observation (Static Zones):** If you notice two river lines overlapping perfectly on the graph above, it indicates a 'Static Zone'—a stretch of the channel where zero energy dissipation occurred.")

            # --- EXCEL / CSV DOWNLOAD ---
            st.markdown("### Export Data")
            csv_data_spatial = convert_df_to_csv(plot_space_df)
            st.download_button(label=f"📄 Download CSV", data=csv_data_spatial, file_name=f"Spatial_{selected_param.replace(' ', '_')}_{space_scen}_{space_wave}m.csv", mime="text/csv")
            
        else:
            st.warning(f"No Spatial data found for {space_scen} at {space_wave}m.")
    
    # ==============================================================
    # TAB 3: MEAN ANALYSIS (WAVE HEIGHT and SEDIMENT TRANSPORT Only)
    # ==============================================================
    with tab3:
        st.subheader("Mean & Variance Analysis")

        if selected_param in {"Significant wave height", "Depth average sediment transport (d. a. sediment transport)"}:
            
            st.markdown(
                f"Mean **{selected_param}** along the river channel for each "
                "offshore wave height condition."
            )
            selected_waves_raw = st.multiselect("Boundary Wave Heights to include",["0.5m", "1m", "2m", "3m"],default=["0.5m", "1m", "2m", "3m"])
            selected_waves = [w.replace("m", "") for w in selected_waves_raw]

            st.markdown("---")
            if not selected_waves:
                st.warning("⚠️ Please select at least one wave height.")
            elif not selected_scenarios:
                st.warning("⚠️ Please select at least one scenario from the sidebar.")
            else:
                y_axis_label = ("Mean Significant Wave Height (cm)" if selected_param == "Significant wave height" else "Mean Sediment Transport (cm³/s/cm)")
                filename = "Wave.csv" if selected_param == "Significant wave height" else "Sediment.csv"
                color_mapping = {
                    'Without Sand Engine and Seagrass': '#1f77b4',
                    'With Sand Engine': '#ff7f0e',
                    'With Seagrass': '#7B2FBE',
                    'With Sand Engine and Seagrass': '#2ca02c'
                }

                # load all data
                # Structure: { wave: { scenario: df } }
                wave_data = {}
                for wave in selected_waves:
                    wave_data[wave] = {}
                    for label in selected_scenarios:
                        folder = os.path.join(scenario_map[label], wave)
                        target_file = os.path.join(folder, filename)
                        try:
                            df = pd.read_csv(target_file, sep=None, engine='python')
                            df['Distance'] = pd.to_numeric(df.iloc[:, 0], errors='coerce')
                            df['Magnitude'] = pd.to_numeric(df.iloc[:, 1], errors='coerce')
                            df = df.dropna(subset=['Distance', 'Magnitude'])
                            df['Scenario'] = label
                            wave_data[wave][label] = df
                        except:
                            pass

                # subplots: one panel per wave height
                n_waves = len(selected_waves)
                fig = make_subplots(
                    rows=1,
                    cols=n_waves,
                    shared_yaxes=True,  # same y axis scale across panels
                    horizontal_spacing=0.02,
                    subplot_titles=[f"Wave height = {w} m<br>(at boundary)" for w in selected_waves]
                )

                # Track which scenarios have been added to legend already
                legend_added = set()
                for col_idx, wave in enumerate(selected_waves, start=1):

                    for label in selected_scenarios:
                        df = wave_data[wave].get(label)
                        if df is None or df.empty:
                            continue

                        show_in_legend = label not in legend_added

                        fig.add_trace(
                            go.Scatter(
                                x=df['Distance'],
                                y=df['Magnitude'],
                                mode='lines',
                                name=label,
                                line=dict(color=color_mapping.get(label, "#000000"), width=2.5),
                                showlegend=show_in_legend,
                                legendgroup=label,  # groups same scenario across panels
                                hovertemplate=f"<b>{label}</b><br>Distance: %{{x}} km<br>Value: %{{y:.4f}}<extra></extra>"
                            ),
                            row=1, col=col_idx
                        )
                        legend_added.add(label)

                   
                    # X axis label only on bottom
                    fig.update_xaxes(
                        title_text="Distance of River Points from Sand Engine (km)" if col_idx == (n_waves // 2 + 1) else "",
                        tickvals=[1, 2, 3, 4, 5, 6, 7],
                        range=[0.5, 7.5],
                        row=1, col=col_idx,
                        showgrid=False
                    )

                # Y axis label on first panel only
                fig.update_yaxes(title_text=y_axis_label, row=1, col=1, showgrid=True)

                fig.update_layout(
                    height=520,
                    plot_bgcolor="white",
                    paper_bgcolor="white",
                    hovermode="x unified",
                    legend=dict(
                        orientation="h",
                        yanchor="top",
                        y=-0.2,
                        xanchor="center",
                        x=0.5,
                        font=dict(size=12)
                    ),
                    margin=dict(t=80, b=100, l=60, r=20)
                )

                st.plotly_chart(fig, use_container_width=True)

        else:
            st.warning("⚠️ Mean data analysis is currently available only for **Significant Wave Height** and **Sediment Transport**. Please select one of these parameters from the sidebar.")

    # ==================================================================
    # TAB 4: PREDICTION MODELS (WAVE HEIGHT and SEDIMENT TRANSPORT Only)
    # ==================================================================
    with tab4:
        st.subheader("ML Surrogate Prediction")
        if selected_param in {"Significant wave height", "Depth average sediment transport (d. a. sediment transport)"}:
            st.markdown(
                "Compare **SVR** and **GPR** surrogate predictions against Delft3D "
                "simulation results along the river channel."
            )
            col1, col2, col3 = st.columns([1,1,1])
            with col1:
                subcol1, subcol2 = st.columns([9,1])
                with subcol1:
                    selected_wave_height = st.selectbox("Boundary Wave Height", ["0.5m","1m","2m","3m"],key="tab4_wave_height")
            with col2:
                st.text("Coastal interventions")
                sub_col1, sub_col2 = st.columns([1,1],vertical_alignment='top')
                with sub_col1:
                    sandengine_present = st.checkbox("Sand Engine", value=True)
                    
                    # if not sandengine_present:
                     #    st.session_state.seagrass_checked = False

                with sub_col2:
                    # seagrass_present = st.checkbox("Seagrass", key="seagrass_checked", disabled=False if sandengine_present else True)
                    seagrass_present = st.checkbox("Seagrass", key="seagrass_checked", value=False)
            
            with col3:
                selected_river_points = st.multiselect("River Points:", ["Point 1","Point 2","Point 3","Point 4","Point 5","Point 6","Point 7"],default=["Point 1","Point 2","Point 3","Point 4","Point 5","Point 6","Point 7"])

            st.markdown("---")

            if not selected_river_points:
                st.warning("⚠️ Please select at least one River Point.")
            else:
                svr_model = joblib.load(os.path.join(base_dir, "SVR_Wave_Height.pkl" if selected_param == "Significant wave height" else "SVR_Sediment_Transport.pkl"))
                gpr_model = joblib.load(os.path.join(base_dir, "GPR_Wave_Height.pkl" if selected_param == "Significant wave height" else "GPR_Sediment_Transport.pkl"))

                input_data = []
                selected_river_points_num = pd.to_numeric([r.replace("Point ", "") for r in selected_river_points])
                for point in selected_river_points_num:
                    input_data.append({
                        'Wave_Height_Category' : float(selected_wave_height.replace("m","")),
                        'Distance_to_SE_km' : float(point),
                        'Sand_Engine_Presence' : 1 if sandengine_present else 0,
                        'Sea_Grass_Presence' : 1 if seagrass_present else 0,
                    })
                input_data = pd.DataFrame(input_data)

                scaler_x = joblib.load(os.path.join(base_dir, "Scaler_x.pkl"))
                input_data_scaled = scaler_x.transform(input_data)
                svr_prediction = svr_model.predict(input_data_scaled)
                gpr_prediction = gpr_model.predict(input_data_scaled)
                
                svr_prediction_df = pd.DataFrame({
                    'River_Points'    : [x for x in selected_river_points_num],
                    'Prediction'      : svr_prediction,
                    'Model'           : 'SVR Prediction'
                })
                gpr_prediction_df = pd.DataFrame({
                    'River_Points'  : [x for x in selected_river_points_num],
                    'Prediction'    : gpr_prediction,
                    'Model'         : 'GPR Prediction'
                })

                # load actual values
                if sandengine_present:
                    if seagrass_present:
                        SE_SG_folder = "With_Sandengine_and_Seagrass"
                    else:
                        SE_SG_folder = "With_Sandengine"
                else:
                    SE_SG_folder = "Without_Sandengine_and_Seagrass"
                
                actual_value_csv = pd.read_csv(
                    os.path.join(base_dir,SE_SG_folder,selected_wave_height.replace("m",""),"Wave.csv" if selected_param == "Significant wave height" else "Sediment.csv"),
                    sep=None, engine='python')
                
                row_index = [int(x)-1 for x in selected_river_points_num]
                actual_value = pd.DataFrame({
                    'River_Points'  : selected_river_points_num,
                    'Prediction'    : pd.to_numeric(actual_value_csv.iloc[row_index,1].values, errors='coerce'),
                    'Model'         : 'Actual value'
                })

                combined_df = pd.concat([svr_prediction_df, gpr_prediction_df, actual_value], ignore_index=True)
                

                # plot
                color_mapping = {
                        'SVR Prediction': '#1f77b4',
                        'GPR Prediction': '#ff7f0e',
                        'Actual value'  : '#2ca02c'
                    }
                fig2 = px.line(combined_df, x='River_Points', y='Prediction', color="Model", color_discrete_map=color_mapping)
                fig2.update_xaxes(
                    tickvals=sorted(combined_df['River_Points'].unique()),
                    ticktext=[f"Point {int(x)}" for x in sorted(combined_df['River_Points'].unique())],
                    range=[combined_df['River_Points'].min() - 0.5, combined_df['River_Points'].max() + 0.5])
                fig2.update_layout(hovermode="x unified", height=500, xaxis_title="River Point", yaxis_title="Wave Height (cm)" if selected_param == "Significant wave height" else "Sediment Transport (cm³/s/cm)", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
                st.plotly_chart(fig2, use_container_width=True)

                df_col1, df_col2 = st.columns([1,1])
                with df_col1:
                    st.subheader("Input variables")
                    st.dataframe(
                        input_data.rename(columns={
                            "Wave_Height_Category": "Wave Height (m)",
                            "Distance_to_SE_km":    "Distance (km)",
                            "Sand_Engine_Presence": "Sand Engine",
                            "Sea_Grass_Presence":   "Seagrass",
                        }),
                        width='stretch', hide_index=True,
                    )
                
                with df_col2:
                    # Target dataframe
                    target_df = pd.DataFrame({
                        'Actual Value'               : actual_value['Prediction'],
                        'SVR Prediction'       : svr_prediction_df['Prediction'],
                        'GPR Prediction'       : gpr_prediction_df['Prediction']
                    })
                    st.subheader("Target variables")
                    st.dataframe(target_df,width='stretch',hide_index=True)


                st.subheader("Model Performance")
                st.markdown(
                    """
                        <style>
                        .dash-card {
                            background: #112233;
                            border: 1px solid rgba(255,255,255,0.08);
                            border-radius: 14px;
                            padding: 1.2rem 1.4rem;
                            margin-bottom: 1rem;
                        }
                        .dash-card h4 { margin-top: 0; }
                        .dash-card table { width: 100%; font-size: 0.85rem; border-collapse: collapse; }
                        .dash-card td { padding: 4px 2px; } </style>
                        """,unsafe_allow_html=True
                )
                mc1, mc2 = st.columns(2)
                with mc1:
                    st.markdown("""
                        <div class="dash-card">
                        <h4 style="color:#4CC9F0">Support Vector Regression (SVR)</h4>
                        <p style="font-size:0.87rem;opacity:0.85;color:#4CC9F0";>
                            SVR model demonstrated excellent prediction capability for both significant
                            wave height and sediment transport. The high testing accuracy presented below
                            indicates that the model successfully generalised the relationship between
                            input variables and target variables: significant wave height and sediment transport.
                        </p>
                        <table>
                            <tr style="color:#4CC9F0"><td>Wave height &nbsp;</td><td><b>R² ≈ 0.99</b></td></tr>
                            <tr style="color:#4CC9F0"><td>Sediment transport</td><td><b>R² ≈ 0.91</b></td></tr>
                        </table>
                        </div>""", unsafe_allow_html=True)
                
                with mc2:
                    st.markdown("""
                        <div class="dash-card">
                        <h4 style="color:#F4A261">Gaussian Process Regression (GPR)</h4>
                        <p style="font-size:0.87rem;opacity:0.85;color:#F4A261">
                            GPR model demonstrated excellent prediction capability for significant wave 
                            height, as shown by the testing results below. However, it struggled to learn
                             the underlying relationship between input variables and sediment transport.
                        </p>
                        <table>
                            <tr style="color:#F4A261"><td>Wave height &nbsp;</td><td><b>R² ≈ 0.99</b></td></tr>
                            <tr style="color:#F4A261"><td>Sediment transport</td><td><b>R² ≈ 0.53</b></td></tr>
                        </table>
                        </div>""", unsafe_allow_html=True)

        else:
            st.warning("⚠️ Prediction models are currently available only for **Significant Wave Height** and **Sediment Transport**. Please select one of these parameters from the sidebar.")

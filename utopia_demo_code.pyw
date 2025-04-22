import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
from sklearn.cluster import KMeans
from transformers import AutoTokenizer, AutoModel
import torch

#sample data used in most of testing, also the default data
sample_data = pd.DataFrame({
    "Movie Title": ["Movie A", "Movie B", "Movie C", "Movie D"],
    "Genre": ["Action, Comedy", "Comedy", "Sci-Fi", "Science Fiction, Action"]
})

#split multi-value cells (aka turn "Action, Comedy" into -> ["Action", "Comedy"] )
def split_multi_values(df, column):
    new_rows = []
    for _, row in df.iterrows():
        genres = str(row[column]).split(',')
        for genre in genres:
            new_row = row.copy()
            new_row[column] = genre.strip()
            new_rows.append(new_row)
    return pd.DataFrame(new_rows)

#cluster similar genres using kmeans cluster and SimCSE sentence embedding
def cluster_genres(genres, n_clusters):
    assert isinstance(n_clusters, int) and n_clusters > 0, "n_clusters must be a positive integer"
    
    #load supervised SimCSE model from Princeton NLP (https://github.com/princeton-nlp/SimCSE) referenced in original paper
    model_name = "princeton-nlp/sup-simcse-bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    #encode each genre string into a sentence embedding using SimCSE
    with torch.no_grad():
        inputs = tokenizer(genres, padding=True, truncation=True, return_tensors="pt")
        outputs = model(**inputs, output_hidden_states=True, return_dict=True)
        embeddings = outputs.pooler_output  # CLS-based sentence embedding

    #cluster the embeddings into the specified amount of clusters
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(embeddings)
    labels = kmeans.labels_

    #create DataFrame of cluster results and assign representative per cluster
    cluster_df = pd.DataFrame({'Original': genres, 'Cluster': labels})
    cluster_df['Representative'] = cluster_df.groupby("Cluster")["Original"].transform('first')
    return dict(zip(cluster_df["Original"], cluster_df["Representative"]))


#main processing function to clean data and display results
def clean_and_display():
    global sample_data, split_df, normalized_df, pivot_df
    
    #ensure proper data is loaded
    if sample_data.empty:
        messagebox.showwarning("No Data", "Please load a CSV file first.")
        return
    if "Genre" not in sample_data.columns:
        messagebox.showerror("Missing Column", "The CSV file must contain a 'Genre' column.")
        return

    try:
        num_clusters = int(cluster_var.get())
        if num_clusters <= 0:
            raise ValueError("Number of clusters must be a positive integer.")
    except Exception as e:
        messagebox.showerror("Invalid Input", f"Invalid number of clusters: {e}")
        return
    
    #split genres into multiple rows
    split_df = split_multi_values(sample_data, "Genre")
    
    #cluster synonymous genres and map them to a normalized genre name
    unique_genres = split_df["Genre"].unique().tolist()
    mapping = cluster_genres(unique_genres, num_clusters)
    split_df["Normalized Genre"] = split_df["Genre"].map(mapping)

    #count how many times each normalized genre appears
    count_df = split_df.pivot_table(index="Normalized Genre", values="Movie Title", aggfunc="count")
    count_df = count_df.rename(columns={"Movie Title": "Count"})

    #build a reverse mapping from representative -> list of grouped synonyms
    reverse_map = {}
    for original, rep in mapping.items():
        reverse_map.setdefault(rep, []).append(original)

    #create a DataFrame from reverse mapping
    synonyms_df = pd.DataFrame({
        "Normalized Genre": list(reverse_map.keys()),
        "Synonyms": [", ".join(sorted(set(syns))) for syns in reverse_map.values()]
    })

    #merge count and synonym info
    pivot_df = count_df.merge(synonyms_df, on="Normalized Genre")

    #show cleaned data and pivot table in UI
    pivot_df = pivot_df.reset_index(drop=True) #removes unecessary 'index' column
    update_treeview(cleaned_tree, split_df)
    update_treeview(pivot_tree, pivot_df)


#update treeview widget in panda frame
def update_treeview(tree, df):
    tree.delete(*tree.get_children()) #clear existing data
    
    #
    tree["columns"] = list(df.columns)
    tree["show"] = "headings"
    
    #set up column headers
    for col in df.columns:
        tree.heading(col, text=col)
        tree.column(col, anchor="center", width=150, stretch=False) #stretching allowed horizontally, but with scrollbars
        
    #insert rows into panda frame
    for _, row in df.iterrows():
        tree.insert("", "end", values=list(row))

#clears all data views
def clear_treeview(tree):
    tree.delete(*tree.get_children())
        
#load in a CSV and reset data views
def load_csv():
    global sample_data
    filepath = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
    if filepath:
        try:
            new_data = pd.read_csv(filepath)
            if "Genre" not in new_data.columns:
                raise ValueError("CSV must contain a 'Genre' column.")
            sample_data = new_data
            update_treeview(original_tree, sample_data)
            clear_treeview(cleaned_tree)
            clear_treeview(pivot_tree)
        except Exception as e:
            messagebox.showerror("Error", f"Could not load file:\n{e}")

#GUI setup
root = tk.Tk()
root.title("Utopia: Pivot Table Assistant Demo By Tristan P.-S.")

#global variable to store user-defined number of clusters (default is 3)
cluster_var = tk.IntVar(value=3)

#load CSV button
ttk.Button(root, text="📂 Load CSV File", command=load_csv).pack(pady=5)

#display original data in table
ttk.Label(root, text="Original Data").pack()
#outer container for spacing and layout
original_outer = ttk.Frame(root)
original_outer.pack(fill="both", expand=True, padx=5, pady=(0, 10))
#treeview with data
original_frame = ttk.Frame(original_outer)
original_frame.pack(fill="both", expand=True)
original_tree = ttk.Treeview(original_frame, height=5)
#scrollbars for treeview
original_scroll_y = ttk.Scrollbar(original_frame, orient="vertical", command=original_tree.yview)
original_scroll_x = ttk.Scrollbar(original_outer, orient="horizontal", command=original_tree.xview)
original_tree.configure(yscrollcommand=original_scroll_y.set, xscrollcommand=original_scroll_x.set)
#main layout
original_tree.grid(row=0, column=0, sticky="nsew")
original_scroll_y.grid(row=0, column=1, sticky="ns")
original_scroll_x.pack(fill="x")
#configure frame to stretch
original_frame.rowconfigure(0, weight=1)
original_frame.columnconfigure(0, weight=1)
#load sample data on startup
update_treeview(original_tree, sample_data)

#have user set clusters and clean data
cluster_frame = ttk.Frame(root)
cluster_frame.pack(pady=10)
#number of clusters label and input
ttk.Label(cluster_frame, text="Number of Clusters:").pack(side="left", padx=(0, 5))
cluster_input = ttk.Spinbox(cluster_frame, from_=1, to=20, textvariable=cluster_var, width=5)
cluster_input.pack(side="left")
#button to run the cleaning process
ttk.Button(cluster_frame, text="Clean and Normalize Data", command=clean_and_display).pack(side="left", padx=(10, 0))

#cleaned data display
ttk.Label(root, text="Cleaned Data").pack()
#outer container
cleaned_outer = ttk.Frame(root)
cleaned_outer.pack(fill="both", expand=True, padx=5, pady=(0, 10))
#treeview with data
cleaned_frame = ttk.Frame(cleaned_outer)
cleaned_frame.pack(fill="both", expand=True)
cleaned_tree = ttk.Treeview(cleaned_frame, height=10)
#scrollbars for treeview
cleaned_scroll_y = ttk.Scrollbar(cleaned_frame, orient="vertical", command=cleaned_tree.yview)
cleaned_scroll_x = ttk.Scrollbar(cleaned_outer, orient="horizontal", command=cleaned_tree.xview)
cleaned_tree.configure(yscrollcommand=cleaned_scroll_y.set, xscrollcommand=cleaned_scroll_x.set)
#main layout
cleaned_tree.grid(row=0, column=0, sticky="nsew")
cleaned_scroll_y.grid(row=0, column=1, sticky="ns")
cleaned_scroll_x.pack(fill="x")
#configure frame to stretch
cleaned_frame.rowconfigure(0, weight=1)
cleaned_frame.columnconfigure(0, weight=1)

#display finalized cleaned data in pivot table (with synonyms)
ttk.Label(root, text="Pivot Table (Genre Counts)").pack()
#outer container
pivot_outer = ttk.Frame(root)
pivot_outer.pack(fill="both", expand=True, padx=5, pady=(0, 10))
#treeview with data
pivot_frame = ttk.Frame(pivot_outer)
pivot_frame.pack(fill="both", expand=True)
pivot_tree = ttk.Treeview(pivot_frame, height=5)
#scrollbars for treeview
pivot_scroll_y = ttk.Scrollbar(pivot_frame, orient="vertical", command=pivot_tree.yview)
pivot_scroll_x = ttk.Scrollbar(pivot_outer, orient="horizontal", command=pivot_tree.xview)
pivot_tree.configure(yscrollcommand=pivot_scroll_y.set, xscrollcommand=pivot_scroll_x.set)
#main layout
pivot_tree.grid(row=0, column=0, sticky="nsew")
pivot_scroll_y.grid(row=0, column=1, sticky="ns")
pivot_scroll_x.pack(fill="x")
#configure frame to stretch
pivot_frame.rowconfigure(0, weight=1)
pivot_frame.columnconfigure(0, weight=1)

#load UI
root.mainloop()
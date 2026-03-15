import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import networkx as nx
from torch_geometric.data import HeteroData
from collections import defaultdict, Counter
import pickle
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class POIGraphAnalyzer:
    def __init__(self, graph_path):
        """Initialize analyzer with POI graph"""
        print("Loading POI Graph...")
        self.data_dict = torch.load(graph_path, weights_only=False)
        self.graph = self.data_dict['graph']
        self.metadata = self.data_dict['metadata']
        
        print("Graph loaded successfully!")
        print(f"Graph structure: {self.graph}")
        
    def basic_statistics(self):
        """Print basic graph statistics"""
        print("\n" + "="*60)
        print("BASIC GRAPH STATISTICS")
        print("="*60)
        
        # Node counts
        print(f"Node types and counts:")
        total_nodes = 0
        for ntype in self.graph.node_types:
            count = self.graph[ntype].num_nodes
            print(f"  {ntype}: {count:,}")
            total_nodes += count
        print(f"  Total nodes: {total_nodes:,}")
        
        # Edge counts  
        print(f"\nEdge types and counts:")
        total_edges = 0
        edge_stats = {}
        for etype in self.graph.edge_types:
            count = self.graph[etype].edge_index.size(1)
            edge_stats[etype] = count
            print(f"  {etype}: {count:,}")
            total_edges += count
        print(f"  Total edges: {total_edges:,}")
        
        # Feature dimensions
        print(f"\nNode feature dimensions:")
        for ntype in self.graph.node_types:
            if hasattr(self.graph[ntype], 'x') and self.graph[ntype].x is not None:
                shape = self.graph[ntype].x.shape
                print(f"  {ntype}: {shape}")
            else:
                print(f"  {ntype}: No features")
                
        # Edge attributes
        print(f"\nEdge attributes:")
        for etype in self.graph.edge_types:
            if hasattr(self.graph[etype], 'edge_attr') and self.graph[etype].edge_attr is not None:
                shape = self.graph[etype].edge_attr.shape
                print(f"  {etype}: {shape}")
            else:
                print(f"  {etype}: No attributes")
        
        return edge_stats
    
    def analyze_degree_distributions(self, edge_stats):
        """Analyze and visualize degree distributions"""
        print("\n" + "="*60)
        print("DEGREE DISTRIBUTION ANALYSIS")
        print("="*60)
        
        # Select main edge types to visualize (skip reverse edges)
        main_edge_types = [e for e in edge_stats.keys() if 'rev_' not in e[1]][:6]
        
        num_plots = len(main_edge_types)
        rows = (num_plots + 2) // 3
        cols = min(3, num_plots)
        
        fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
        if num_plots == 1:
            axes = [axes]
        else:
            axes = axes.flatten() if num_plots > 1 else [axes]
        
        degree_stats = {}
        
        for idx, etype in enumerate(main_edge_types):
            edge_index = self.graph[etype].edge_index
            src, dst = edge_index[0].cpu().numpy(), edge_index[1].cpu().numpy()
            
            # Calculate degrees
            src_nodes = self.graph[etype[0]].num_nodes
            dst_nodes = self.graph[etype[2]].num_nodes
            
            src_deg = np.bincount(src, minlength=src_nodes)
            dst_deg = np.bincount(dst, minlength=dst_nodes)
            
            degree_stats[etype] = {
                'src_mean': src_deg.mean(),
                'src_std': src_deg.std(),
                'src_max': src_deg.max(),
                'dst_mean': dst_deg.mean(),
                'dst_std': dst_deg.std(),
                'dst_max': dst_deg.max()
            }
            
            # Plot
            ax = axes[idx]
            ax.hist(src_deg[src_deg > 0], bins=min(30, len(np.unique(src_deg))), 
                   alpha=0.7, label=f"{etype[0]} out-degree", density=True)
            ax.hist(dst_deg[dst_deg > 0], bins=min(30, len(np.unique(dst_deg))), 
                   alpha=0.7, label=f"{etype[2]} in-degree", density=True)
            ax.set_title(f"Degree Distribution: {etype[1]}")
            ax.set_xlabel("Degree")
            ax.set_ylabel("Density")
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            print(f"\n{etype}:")
            print(f"  Source ({etype[0]}) - mean: {src_deg.mean():.2f}, std: {src_deg.std():.2f}, max: {src_deg.max()}")
            print(f"  Target ({etype[2]}) - mean: {dst_deg.mean():.2f}, std: {dst_deg.std():.2f}, max: {dst_deg.max()}")
        
        # Hide unused subplots
        for idx in range(num_plots, len(axes)):
            axes[idx].set_visible(False)
            
        plt.tight_layout()
        plt.savefig('degree_distributions.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return degree_stats
    
    def analyze_node_features(self):
        """Analyze node feature distributions"""
        print("\n" + "="*60)
        print("NODE FEATURE ANALYSIS")
        print("="*60)
        
        feature_stats = {}
        
        # User features analysis (Age + Gender only)
        if 'user' in self.graph.node_types and hasattr(self.graph['user'], 'x'):
            user_features = self.graph['user'].x.cpu().numpy()
            print(f"\nUser Features Analysis:")
            print(f"  Shape: {user_features.shape}")
            print(f"  Mean: {user_features.mean(axis=0)}")
            print(f"  Std: {user_features.std(axis=0)}")
            
            # Plot user feature distributions
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            
            # Age groups (first 5 features)
            age_features = user_features[:, :5]
            age_labels = ['18-25', '26-35', '36-45', '46-55', '56+']
            age_counts = age_features.sum(axis=0)
            
            axes[0].bar(age_labels, age_counts)
            axes[0].set_title('User Age Distribution')
            axes[0].set_ylabel('Count')
            axes[0].tick_params(axis='x', rotation=45)
            
            # Gender (next 3 features)
            gender_features = user_features[:, 5:8]
            gender_labels = ['Male', 'Female', 'Other']
            gender_counts = gender_features.sum(axis=0)
            
            axes[1].bar(gender_labels, gender_counts)
            axes[1].set_title('User Gender Distribution')
            axes[1].set_ylabel('Count')
            
            plt.tight_layout()
            plt.savefig('user_features_analysis.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            feature_stats['user'] = {
                'shape': user_features.shape,
                'mean': user_features.mean(axis=0),
                'std': user_features.std(axis=0),
                'age_distribution': age_counts,
                'gender_distribution': gender_counts
            }
        
        # POI features analysis (Sentiment score only)
        if 'poi' in self.graph.node_types and hasattr(self.graph['poi'], 'x'):
            poi_features = self.graph['poi'].x.cpu().numpy()
            print(f"\nPOI Features Analysis:")
            print(f"  Shape: {poi_features.shape}")
            print(f"  Sentiment Score - Mean: {poi_features.mean():.3f}, Std: {poi_features.std():.3f}")
            print(f"  Sentiment Score - Min: {poi_features.min():.3f}, Max: {poi_features.max():.3f}")
            
            # Plot POI sentiment distribution
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            
            # Histogram
            axes[0].hist(poi_features.flatten(), bins=30, edgecolor='black', alpha=0.7)
            axes[0].set_title('POI Sentiment Score Distribution')
            axes[0].set_xlabel('Sentiment Score')
            axes[0].set_ylabel('Frequency')
            axes[0].grid(True, alpha=0.3)
            
            # Box plot
            axes[1].boxplot(poi_features.flatten())
            axes[1].set_title('POI Sentiment Score Box Plot')
            axes[1].set_ylabel('Sentiment Score')
            axes[1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig('poi_features_analysis.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            feature_stats['poi'] = {
                'shape': poi_features.shape,
                'mean': poi_features.mean(),
                'std': poi_features.std(),
                'min': poi_features.min(),
                'max': poi_features.max()
            }
        
        # SensoryAttribute features analysis
        if 'sensory_attr' in self.graph.node_types and hasattr(self.graph['sensory_attr'], 'x'):
            sensory_features = self.graph['sensory_attr'].x.cpu().numpy()
            print(f"\nSensoryAttribute Features Analysis:")
            print(f"  Shape: {sensory_features.shape}")
            print(f"  Number of unique sensory nodes: {len(sensory_features)}")
            
            # Type one-hot is first 5 features, value is last feature
            sensory_types = sensory_features[:, :5]
            sensory_values = sensory_features[:, 5]
            
            # Count by type
            type_labels = ['crowd', 'noise', 'space', 'brightness', 'visual_clutter']
            type_counts = sensory_types.sum(axis=0)
            
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            
            # Type distribution
            axes[0].bar(type_labels, type_counts)
            axes[0].set_title('Shared SensoryAttribute Node Distribution by Type')
            axes[0].set_ylabel('Count')
            axes[0].tick_params(axis='x', rotation=45)
            
            # Value distribution
            axes[1].hist(sensory_values, bins=30, edgecolor='black', alpha=0.7)
            axes[1].set_title('SensoryAttribute Value Distribution')
            axes[1].set_xlabel('Attribute Value')
            axes[1].set_ylabel('Frequency')
            axes[1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig('sensory_attr_features_analysis.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            feature_stats['sensory_attr'] = {
                'shape': sensory_features.shape,
                'type_distribution': dict(zip(type_labels, type_counts)),
                'value_mean': sensory_values.mean(),
                'value_std': sensory_values.std()
            }
        
        # OtherAttribute features analysis (NEW: with polarity)
        if 'other_attr' in self.graph.node_types and hasattr(self.graph['other_attr'], 'x'):
            other_features = self.graph['other_attr'].x.cpu().numpy()
            print(f"\nOtherAttribute Features Analysis (VADER Sentiment):")
            print(f"  Shape: {other_features.shape}")
            print(f"  Number of other attribute nodes: {len(other_features)}")
            
            # Features: [7 type one-hot] + [3 polarity one-hot] + [sentiment value]
            type_onehot = other_features[:, :7]
            polarity_onehot = other_features[:, 7:10]
            sentiment_values = other_features[:, 10]
            
            type_labels = ['food', 'staff', 'service', 'ambiance', 'cleanliness', 'atmosphere', 'facilities']
            polarity_labels = ['Negative', 'Neutral', 'Positive']
            
            # Count by type and polarity
            type_counts = type_onehot.sum(axis=0)
            polarity_counts = polarity_onehot.sum(axis=0)
            
            fig, axes = plt.subplots(1, 3, figsize=(18, 5))
            
            # Type distribution
            axes[0].bar(type_labels, type_counts, color='#95E1D3', edgecolor='black', alpha=0.8)
            axes[0].set_title('OtherAttribute Types with VADER Sentiment')
            axes[0].set_ylabel('Number of Nodes')
            axes[0].tick_params(axis='x', rotation=45)
            axes[0].grid(True, alpha=0.3, axis='y')
            
            # Polarity distribution
            colors_polarity = ['#FF6B6B', '#FFE66D', '#4ECDC4']
            axes[1].bar(polarity_labels, polarity_counts, color=colors_polarity, edgecolor='black', alpha=0.8)
            axes[1].set_title('Sentiment Polarity Distribution')
            axes[1].set_ylabel('Number of Nodes')
            axes[1].grid(True, alpha=0.3, axis='y')
            
            # Sentiment value distribution
            axes[2].hist(sentiment_values, bins=20, edgecolor='black', alpha=0.7, color='#95E1D3')
            axes[2].set_title('VADER Sentiment Score Distribution')
            axes[2].set_xlabel('Sentiment Score (-1 to 1)')
            axes[2].set_ylabel('Frequency')
            axes[2].axvline(x=0, color='red', linestyle='--', alpha=0.5, label='Neutral')
            axes[2].legend()
            axes[2].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig('other_attr_features_analysis.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            print(f"  Sentiment statistics:")
            print(f"    Mean: {sentiment_values.mean():.3f}")
            print(f"    Std: {sentiment_values.std():.3f}")
            print(f"    Min: {sentiment_values.min():.3f}")
            print(f"    Max: {sentiment_values.max():.3f}")
            
            feature_stats['other_attr'] = {
                'shape': other_features.shape,
                'type_distribution': dict(zip(type_labels, type_counts)),
                'polarity_distribution': dict(zip(polarity_labels, polarity_counts)),
                'sentiment_mean': sentiment_values.mean(),
                'sentiment_std': sentiment_values.std(),
                'sentiment_min': sentiment_values.min(),
                'sentiment_max': sentiment_values.max()
            }
        
        # Category features analysis
        if 'category' in self.graph.node_types and hasattr(self.graph['category'], 'x'):
            category_features = self.graph['category'].x.cpu().numpy()
            print(f"\nCategory Features Analysis:")
            print(f"  Shape: {category_features.shape}")
            print(f"  Number of categories: {len(category_features)}")
            
            # Get category names from metadata
            category_names = sorted(self.metadata['category_to_idx'].keys(), 
                                   key=lambda x: self.metadata['category_to_idx'][x])
            
            # Plot as horizontal bar chart for better readability
            plt.figure(figsize=(10, max(6, len(category_names) * 0.4)))
            y_pos = np.arange(len(category_names))
            
            # Count POIs per category (from edges)
            if ('poi', 'belongs_to', 'category') in self.graph.edge_types:
                poi_category_edges = self.graph[('poi', 'belongs_to', 'category')].edge_index
                category_counts = np.bincount(poi_category_edges[1].cpu().numpy(), 
                                             minlength=len(category_names))
            else:
                category_counts = np.zeros(len(category_names))
            
            plt.barh(y_pos, category_counts, color='#F6BB4F', edgecolor='black', alpha=0.8)
            plt.yticks(y_pos, category_names)
            plt.xlabel('Number of POIs')
            plt.title('POI Distribution by Category')
            plt.grid(True, alpha=0.3, axis='x')
            plt.tight_layout()
            plt.savefig('category_features_analysis.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            feature_stats['category'] = {
                'shape': category_features.shape,
                'categories': category_names,
                'poi_counts': dict(zip(category_names, category_counts))
            }
        
        return feature_stats
    
    def analyze_edge_attributes(self):
        """Analyze edge attribute distributions"""
        print("\n" + "="*60)
        print("EDGE ATTRIBUTE ANALYSIS")
        print("="*60)
        
        edge_attr_stats = {}
        
        # Ratings analysis
        if ('user', 'rates', 'poi') in self.graph.edge_types:
            if hasattr(self.graph[('user', 'rates', 'poi')], 'edge_attr'):
                ratings = self.graph[('user', 'rates', 'poi')].edge_attr.cpu().numpy().flatten()
                
                if len(ratings) > 0:  # Check if not empty
                    print(f"\nRating Statistics:")
                    print(f"  Count: {len(ratings)}")
                    print(f"  Mean: {ratings.mean():.3f}")
                    print(f"  Std: {ratings.std():.3f}")
                    print(f"  Min: {ratings.min():.3f}")
                    print(f"  Max: {ratings.max():.3f}")
                    print(f"  Median: {np.median(ratings):.3f}")
                    
                    # Plot rating distribution
                    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
                    
                    axes[0].hist(ratings, bins=20, edgecolor='black', alpha=0.7)
                    axes[0].set_title('Rating Distribution')
                    axes[0].set_xlabel('Rating')
                    axes[0].set_ylabel('Frequency')
                    axes[0].grid(True, alpha=0.3)
                    
                    # Rating distribution by integer values
                    rating_counts = Counter(ratings)
                    unique_ratings = sorted(rating_counts.keys())
                    counts = [rating_counts[r] for r in unique_ratings]
                    
                    axes[1].bar(unique_ratings, counts, alpha=0.7, edgecolor='black')
                    axes[1].set_title('Rating Value Counts')
                    axes[1].set_xlabel('Rating')
                    axes[1].set_ylabel('Count')
                    axes[1].grid(True, alpha=0.3)
                    
                    plt.tight_layout()
                    plt.savefig('rating_distribution.png', dpi=300, bbox_inches='tight')
                    plt.show()
                    
                    edge_attr_stats['ratings'] = {
                        'count': len(ratings),
                        'mean': ratings.mean(),
                        'std': ratings.std(),
                        'min': ratings.min(),
                        'max': ratings.max(),
                        'distribution': dict(rating_counts)
                    }
                else:
                    print(f"\nRating Statistics:")
                    print(f"  No rating edge attributes found")
                    edge_attr_stats['ratings'] = {'count': 0}
        
        # Category preferences analysis
        if ('user', 'prefers', 'category') in self.graph.edge_types:
            if hasattr(self.graph[('user', 'prefers', 'category')], 'edge_attr'):
                preferences = self.graph[('user', 'prefers', 'category')].edge_attr.cpu().numpy().flatten()
                
                if len(preferences) > 0:  # Check if not empty
                    print(f"\nCategory Preference Statistics:")
                    print(f"  Count: {len(preferences)}")
                    print(f"  Mean: {preferences.mean():.3f}")
                    print(f"  Std: {preferences.std():.3f}")
                    print(f"  Min: {preferences.min():.3f}")
                    print(f"  Max: {preferences.max():.3f}")
                    
                    plt.figure(figsize=(8, 5))
                    plt.hist(preferences, bins=15, edgecolor='black', alpha=0.7)
                    plt.title('Category Preference Distribution')
                    plt.xlabel('Preference Score')
                    plt.ylabel('Frequency')
                    plt.grid(True, alpha=0.3)
                    plt.savefig('preference_distribution.png', dpi=300, bbox_inches='tight')
                    plt.show()
                    
                    edge_attr_stats['preferences'] = {
                        'count': len(preferences),
                        'mean': preferences.mean(),
                        'std': preferences.std(),
                        'min': preferences.min(),
                        'max': preferences.max()
                    }
                else:
                    print(f"\nCategory Preference Statistics:")
                    print(f"  No preference edge attributes found")
                    edge_attr_stats['preferences'] = {'count': 0}
        
        # NEW: Other attribute sentiment edge weights
        if ('poi', 'has_other_attribute', 'other_attr') in self.graph.edge_types:
            if hasattr(self.graph[('poi', 'has_other_attribute', 'other_attr')], 'edge_attr'):
                sentiments = self.graph[('poi', 'has_other_attribute', 'other_attr')].edge_attr.cpu().numpy().flatten()
                
                if len(sentiments) > 0:  # Check if not empty
                    print(f"\nOther Attribute Sentiment Statistics (VADER):")
                    print(f"  Count: {len(sentiments)}")
                    print(f"  Mean: {sentiments.mean():.3f}")
                    print(f"  Std: {sentiments.std():.3f}")
                    print(f"  Min: {sentiments.min():.3f}")
                    print(f"  Max: {sentiments.max():.3f}")
                    
                    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
                    
                    # Histogram
                    axes[0].hist(sentiments, bins=30, edgecolor='black', alpha=0.7, color='#95E1D3')
                    axes[0].axvline(x=0, color='red', linestyle='--', alpha=0.5, label='Neutral')
                    axes[0].set_title('VADER Sentiment Edge Weight Distribution')
                    axes[0].set_xlabel('Sentiment Score (-1 to 1)')
                    axes[0].set_ylabel('Frequency')
                    axes[0].legend()
                    axes[0].grid(True, alpha=0.3)
                    
                    # Box plot by polarity
                    negative = sentiments[sentiments <= -0.3]
                    neutral = sentiments[(sentiments > -0.3) & (sentiments < 0.3)]
                    positive = sentiments[sentiments >= 0.3]
                    
                    # Only plot non-empty groups
                    data_to_plot = []
                    labels_to_plot = []
                    if len(negative) > 0:
                        data_to_plot.append(negative)
                        labels_to_plot.append('Negative')
                    if len(neutral) > 0:
                        data_to_plot.append(neutral)
                        labels_to_plot.append('Neutral')
                    if len(positive) > 0:
                        data_to_plot.append(positive)
                        labels_to_plot.append('Positive')
                    
                    if data_to_plot:
                        axes[1].boxplot(data_to_plot, labels=labels_to_plot)
                        axes[1].set_title('Sentiment Distribution by Polarity Bucket')
                        axes[1].set_ylabel('Sentiment Score')
                        axes[1].grid(True, alpha=0.3, axis='y')
                    
                    plt.tight_layout()
                    plt.savefig('other_attr_sentiment_distribution.png', dpi=300, bbox_inches='tight')
                    plt.show()
                    
                    edge_attr_stats['other_attr_sentiments'] = {
                        'count': len(sentiments),
                        'mean': sentiments.mean(),
                        'std': sentiments.std(),
                        'min': sentiments.min(),
                        'max': sentiments.max(),
                        'negative_count': len(negative),
                        'neutral_count': len(neutral),
                        'positive_count': len(positive)
                    }
                else:
                    print(f"\nOther Attribute Sentiment Statistics:")
                    print(f"  No sentiment edge attributes found")
                    edge_attr_stats['other_attr_sentiments'] = {'count': 0}
        
        return edge_attr_stats
    
    def analyze_connectivity(self):
        """Analyze graph connectivity patterns"""
        print("\n" + "="*60)
        print("CONNECTIVITY ANALYSIS")
        print("="*60)
        
        connectivity_stats = {}
        
        # Analyze each edge type separately
        for etype in self.graph.edge_types:
            if 'rev_' in etype[1]:  # Skip reverse edges to avoid duplication
                continue
                
            edge_index = self.graph[etype].edge_index
            src, dst = edge_index[0].cpu().numpy(), edge_index[1].cpu().numpy()
            
            # Create NetworkX graph for connectivity analysis
            G = nx.Graph()
            edges = list(zip(src, dst))
            G.add_edges_from(edges)
            
            # Calculate connectivity metrics
            num_nodes = G.number_of_nodes()
            num_edges = G.number_of_edges()
            num_components = nx.number_connected_components(G)
            
            if num_components > 0:
                largest_cc = max(nx.connected_components(G), key=len)
                largest_cc_size = len(largest_cc)
                connectivity_ratio = largest_cc_size / num_nodes if num_nodes > 0 else 0
            else:
                largest_cc_size = 0
                connectivity_ratio = 0
            
            connectivity_stats[etype] = {
                'nodes': num_nodes,
                'edges': num_edges,
                'components': num_components,
                'largest_component': largest_cc_size,
                'connectivity_ratio': connectivity_ratio
            }
            
            print(f"\n{etype}:")
            print(f"  Nodes in subgraph: {num_nodes:,}")
            print(f"  Edges: {num_edges:,}")
            print(f"  Connected components: {num_components}")
            print(f"  Largest component: {largest_cc_size:,} ({connectivity_ratio:.1%})")
        
        return connectivity_stats
    
    def analyze_shared_nodes(self):
        """Analyze shared sensory and other attribute nodes"""
        print("\n" + "="*60)
        print("SHARED NODE ANALYSIS")
        print("="*60)
        
        shared_stats = {}
        
        # Analyze SensoryAttribute sharing
        if ('user', 'has_sensory_preference', 'sensory_attr') in self.graph.edge_types and \
           ('poi', 'has_sensory_attribute', 'sensory_attr') in self.graph.edge_types:
            
            user_sensory_edges = self.graph[('user', 'has_sensory_preference', 'sensory_attr')].edge_index
            poi_sensory_edges = self.graph[('poi', 'has_sensory_attribute', 'sensory_attr')].edge_index
            
            # Get unique sensory nodes connected to users and POIs
            user_sensory_nodes = set(user_sensory_edges[1].cpu().numpy())
            poi_sensory_nodes = set(poi_sensory_edges[1].cpu().numpy())
            
            # Find shared nodes
            shared_sensory = user_sensory_nodes & poi_sensory_nodes
            user_only = user_sensory_nodes - poi_sensory_nodes
            poi_only = poi_sensory_nodes - user_sensory_nodes
            
            print(f"\nSensoryAttribute Sharing Analysis:")
            print(f"  Total unique sensory nodes: {self.graph['sensory_attr'].num_nodes}")
            print(f"  Nodes connected to users: {len(user_sensory_nodes)}")
            print(f"  Nodes connected to POIs: {len(poi_sensory_nodes)}")
            print(f"  Shared nodes (connected to both): {len(shared_sensory)}")
            print(f"  User-only nodes: {len(user_only)}")
            print(f"  POI-only nodes: {len(poi_only)}")
            print(f"  Sharing ratio: {len(shared_sensory) / self.graph['sensory_attr'].num_nodes * 100:.1f}%")
            
            # Visualize
            fig, ax = plt.subplots(figsize=(8, 6))
            categories = ['Shared\n(User & POI)', 'User Only', 'POI Only']
            counts = [len(shared_sensory), len(user_only), len(poi_only)]
            colors = ['#4ECDC4', '#FF6B6B', '#FFE66D']
            
            ax.bar(categories, counts, color=colors, edgecolor='black', alpha=0.8)
            ax.set_title('SensoryAttribute Node Sharing Pattern')
            ax.set_ylabel('Number of Nodes')
            ax.grid(True, alpha=0.3, axis='y')
            
            for i, (cat, count) in enumerate(zip(categories, counts)):
                ax.text(i, count + max(counts)*0.02, str(count), 
                       ha='center', va='bottom', fontsize=12, fontweight='bold')
            
            plt.tight_layout()
            plt.savefig('sensory_node_sharing.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            shared_stats['sensory_attr'] = {
                'total_nodes': self.graph['sensory_attr'].num_nodes,
                'user_connected': len(user_sensory_nodes),
                'poi_connected': len(poi_sensory_nodes),
                'shared': len(shared_sensory),
                'user_only': len(user_only),
                'poi_only': len(poi_only),
                'sharing_ratio': len(shared_sensory) / self.graph['sensory_attr'].num_nodes if self.graph['sensory_attr'].num_nodes > 0 else 0
            }
        
        # Analyze OtherAttribute connections (NEW: with polarity breakdown)
        if ('poi', 'has_other_attribute', 'other_attr') in self.graph.edge_types:
            poi_other_edges = self.graph[('poi', 'has_other_attribute', 'other_attr')].edge_index
            
            # Count POIs connected to each other attribute
            other_attr_degrees = np.bincount(poi_other_edges[1].cpu().numpy(), 
                                            minlength=self.graph['other_attr'].num_nodes)
            
            # Get polarity breakdown from metadata
            other_attr_to_idx = self.metadata.get('other_attr_to_idx', {})
            polarity_breakdown = defaultdict(lambda: {'Negative': 0, 'Neutral': 0, 'Positive': 0})
            
            for (attr_type, polarity), idx in other_attr_to_idx.items():
                polarity_breakdown[attr_type][polarity] = other_attr_degrees[idx]
            
            print(f"\nOtherAttribute Connection Statistics (VADER Polarity):")
            print(f"  Total other attribute nodes: {self.graph['other_attr'].num_nodes}")
            
            attr_types = ['food', 'staff', 'service', 'ambiance', 'cleanliness', 'atmosphere', 'facilities']
            
            for attr_type in attr_types:
                if attr_type in polarity_breakdown:
                    breakdown = polarity_breakdown[attr_type]
                    total = sum(breakdown.values())
                    print(f"  {attr_type}: {total} POIs total")
                    print(f"    Positive: {breakdown['Positive']}, Neutral: {breakdown['Neutral']}, Negative: {breakdown['Negative']}")
            
            # Visualize with stacked bar chart
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Prepare data for stacked bars
            x_labels = []
            positive_counts = []
            neutral_counts = []
            negative_counts = []
            
            for attr_type in attr_types:
                if attr_type in polarity_breakdown:
                    x_labels.append(attr_type)
                    breakdown = polarity_breakdown[attr_type]
                    positive_counts.append(breakdown['Positive'])
                    neutral_counts.append(breakdown['Neutral'])
                    negative_counts.append(breakdown['Negative'])
            
            x = np.arange(len(x_labels))
            width = 0.6
            
            p1 = ax.bar(x, positive_counts, width, label='Positive', color='#4ECDC4', edgecolor='black')
            p2 = ax.bar(x, neutral_counts, width, bottom=positive_counts, label='Neutral', color='#FFE66D', edgecolor='black')
            p3 = ax.bar(x, negative_counts, width, bottom=np.array(positive_counts)+np.array(neutral_counts), 
                       label='Negative', color='#FF6B6B', edgecolor='black')
            
            ax.set_title('POI Connections to OtherAttribute Nodes (VADER Sentiment Polarity)')
            ax.set_xlabel('Attribute Type')
            ax.set_ylabel('Number of POIs Connected')
            ax.set_xticks(x)
            ax.set_xticklabels(x_labels, rotation=45, ha='right')
            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')
            
            plt.tight_layout()
            plt.savefig('other_attr_connections.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            shared_stats['other_attr'] = {
                'total_nodes': self.graph['other_attr'].num_nodes,
                'polarity_breakdown': dict(polarity_breakdown)
            }
        
        return shared_stats
    
    def sparsity_analysis(self):
        """Analyze graph sparsity"""
        print("\n" + "="*60)
        print("SPARSITY ANALYSIS")
        print("="*60)
        
        sparsity_stats = {}
        
        for etype in self.graph.edge_types:
            if 'rev_' in etype[1]:  # Skip reverse edges
                continue
                
            src_nodes = self.graph[etype[0]].num_nodes
            dst_nodes = self.graph[etype[2]].num_nodes
            actual_edges = self.graph[etype].edge_index.size(1)
            
            # Calculate maximum possible edges
            if etype[0] == etype[2]:  # Same node type
                max_edges = src_nodes * (src_nodes - 1) // 2
            else:
                max_edges = src_nodes * dst_nodes
            
            sparsity = 1 - (actual_edges / max_edges) if max_edges > 0 else 0
            density = actual_edges / max_edges if max_edges > 0 else 0
            
            sparsity_stats[etype] = {
                'actual_edges': actual_edges,
                'max_possible_edges': max_edges,
                'density': density,
                'sparsity': sparsity
            }
            
            print(f"\n{etype}:")
            print(f"  Actual edges: {actual_edges:,}")
            print(f"  Max possible edges: {max_edges:,}")
            print(f"  Density: {density:.6f} ({density*100:.4f}%)")
            print(f"  Sparsity: {sparsity:.6f} ({sparsity*100:.4f}%)")
        
        return sparsity_stats
    
    def data_quality_assessment(self):
        """Assess data quality metrics"""
        print("\n" + "="*60)
        print("DATA QUALITY ASSESSMENT")
        print("="*60)
        
        stats = self.metadata.get('statistics', {})
        
        print(f"Data Processing Results:")
        print(f"  Reviews processed: {stats.get('total_reviews_processed', 'N/A')}")
        print(f"  POIs with sentiment: {stats.get('pois_with_sentiment', 'N/A')}")
        print(f"  POIs with sensory attributes: {stats.get('pois_with_sensory', 'N/A')}")
        print(f"  POIs with other attributes (VADER): {stats.get('pois_with_other_attrs', 'N/A')}")
        print(f"  Shared sensory nodes: {stats.get('shared_sensory_nodes', 'N/A')}")
        print(f"  Shared other nodes (with polarity): {stats.get('shared_other_nodes', 'N/A')}")
        
        # Calculate coverage ratios
        total_pois = self.graph['poi'].num_nodes
        sentiment_coverage = stats.get('pois_with_sentiment', 0) / total_pois * 100
        sensory_coverage = stats.get('pois_with_sensory', 0) / total_pois * 100
        other_coverage = stats.get('pois_with_other_attrs', 0) / total_pois * 100
        
        print(f"\nAttribute Coverage:")
        print(f"  Sentiment score coverage: {sentiment_coverage:.1f}%")
        print(f"  Sensory attribute coverage: {sensory_coverage:.1f}%")
        print(f"  Other attribute coverage (VADER): {other_coverage:.1f}%")
        
        # Feature completeness
        feature_completeness = {}
        
        for ntype in ['user', 'poi', 'sensory_attr', 'other_attr', 'category']:
            if ntype in self.graph.node_types and hasattr(self.graph[ntype], 'x'):
                features = self.graph[ntype].x.cpu().numpy()
                missing = np.isnan(features).sum() / features.size * 100
                feature_completeness[ntype] = 100 - missing
        
        print(f"\nFeature Completeness:")
        for ntype, completeness in feature_completeness.items():
            print(f"  {ntype}: {completeness:.2f}%")
        
        return {
            'coverage': {
                'sentiment': sentiment_coverage,
                'sensory': sensory_coverage,
                'other': other_coverage
            },
            'completeness': feature_completeness,
            'processing_stats': stats
        }
    
    def preprocess_graph(self):
        """Apply preprocessing to the graph"""
        print("\n" + "="*60)
        print("GRAPH PREPROCESSING")
        print("="*60)
        
        preprocessed_data = self.graph.clone()
        mappings = {}
        
        # 1. Normalize POI sentiment scores
        print("Normalizing POI sentiment scores...")
        
        if 'poi' in preprocessed_data.node_types and hasattr(preprocessed_data['poi'], 'x'):
            scaler = StandardScaler()
            poi_sentiment = preprocessed_data['poi'].x.cpu().numpy()
            preprocessed_data['poi'].x = torch.tensor(
                scaler.fit_transform(poi_sentiment), dtype=torch.float
            )
            mappings['poi_sentiment_scaler'] = scaler
            print(f"  Normalized POI sentiment: {poi_sentiment.shape}")
        
        # 2. SensoryAttribute values
        print("Normalizing SensoryAttribute values...")
        
        if 'sensory_attr' in preprocessed_data.node_types and hasattr(preprocessed_data['sensory_attr'], 'x'):
            sensory_features = preprocessed_data['sensory_attr'].x.cpu().numpy()
            sensory_types = sensory_features[:, :5]
            sensory_values = sensory_features[:, 5:6]
            
            scaler = StandardScaler()
            normalized_values = scaler.fit_transform(sensory_values)
            
            preprocessed_data['sensory_attr'].x = torch.tensor(
                np.concatenate([sensory_types, normalized_values], axis=1), dtype=torch.float
            )
            mappings['sensory_value_scaler'] = scaler
            print(f"  Normalized sensory values: {sensory_values.shape}")
        
        # 3. User features (one-hot, no normalization)
        print("User features (one-hot encoded, no normalization needed)")
        
        # 4. OtherAttribute features (NEW: normalize sentiment values)
        print("Normalizing OtherAttribute sentiment values...")
        
        if 'other_attr' in preprocessed_data.node_types and hasattr(preprocessed_data['other_attr'], 'x'):
            other_features = preprocessed_data['other_attr'].x.cpu().numpy()
            # Features: [7 type] + [3 polarity] + [1 sentiment]
            type_polarity = other_features[:, :10]
            sentiment_values = other_features[:, 10:11]
            
            scaler = StandardScaler()
            normalized_sentiment = scaler.fit_transform(sentiment_values)
            
            preprocessed_data['other_attr'].x = torch.tensor(
                np.concatenate([type_polarity, normalized_sentiment], axis=1), dtype=torch.float
            )
            mappings['other_attr_sentiment_scaler'] = scaler
            print(f"  Normalized other attribute sentiments: {sentiment_values.shape}")
        
        # 5. Normalize edge attributes
        print("Normalizing edge attributes...")
        
        edge_types_to_normalize = [
            ('user', 'rates', 'poi'), 
            ('user', 'prefers', 'category'), 
            ('user', 'has_sensory_preference', 'sensory_attr'),
            ('poi', 'has_other_attribute', 'other_attr')
        ]
        
        for etype in edge_types_to_normalize:
            if (etype in preprocessed_data.edge_types and 
                hasattr(preprocessed_data[etype], 'edge_attr') and
                preprocessed_data[etype].edge_attr is not None):
                
                edge_attr = preprocessed_data[etype].edge_attr.cpu().numpy()
                
                # Only normalize if not empty
                if edge_attr.shape[0] > 0:
                    scaler = MinMaxScaler()
                    preprocessed_data[etype].edge_attr = torch.tensor(
                        scaler.fit_transform(edge_attr), dtype=torch.float
                    )
                    mappings[f"{etype[1]}_edge_scaler"] = scaler
                    print(f"  Normalized {etype} edge attributes: {edge_attr.shape}")
                else:
                    print(f"  Skipped {etype} edge attributes (empty)")
        
        # 6. Copy normalization to reverse edges
        print("Copying normalization to reverse edges...")
        
        reverse_edge_mapping = {
            ('user', 'rates', 'poi'): ('poi', 'rev_rates', 'user'),
            ('user', 'prefers', 'category'): ('category', 'rev_prefers', 'user'),
            ('user', 'has_sensory_preference', 'sensory_attr'): ('sensory_attr', 'rev_has_sensory_preference', 'user'),
            ('poi', 'has_other_attribute', 'other_attr'): ('other_attr', 'rev_has_other_attribute', 'poi')
        }
        
        for original_etype, reverse_etype in reverse_edge_mapping.items():
            if (original_etype in preprocessed_data.edge_types and 
                reverse_etype in preprocessed_data.edge_types and
                hasattr(preprocessed_data[original_etype], 'edge_attr') and
                preprocessed_data[original_etype].edge_attr is not None):
                
                preprocessed_data[reverse_etype].edge_attr = preprocessed_data[original_etype].edge_attr
                print(f"  Copied attributes to {reverse_etype}")
        
        # 7. Clean up empty edges
        print("Cleaning up empty edges...")
        
        to_delete = []
        for etype in list(preprocessed_data.edge_types):
            edge_store = preprocessed_data[etype]
            edge_index = getattr(edge_store, "edge_index", None)
            if edge_index is None or edge_index.numel() == 0:
                to_delete.append(etype)
        
        for etype in to_delete:
            del preprocessed_data[etype]
            print(f"  Removed empty edge type: {etype}")
        
        # Save preprocessing mappings
        with open("poi_preprocessing_mappings.pkl", "wb") as f:
            pickle.dump(mappings, f)
        
        print(f"Preprocessing mappings saved to poi_preprocessing_mappings.pkl")
        print("Preprocessing completed!")
        
        return preprocessed_data, mappings
    
    def comprehensive_analysis(self):
        """Run complete analysis pipeline"""
        print("Starting Comprehensive POI Graph Analysis...")
        print("="*80)
        
        # Basic statistics
        edge_stats = self.basic_statistics()
        
        # Degree distributions
        degree_stats = self.analyze_degree_distributions(edge_stats)
        
        # Node features
        feature_stats = self.analyze_node_features()
        
        # Edge attributes
        edge_attr_stats = self.analyze_edge_attributes()
        
        # Shared nodes analysis
        shared_stats = self.analyze_shared_nodes()
        
        # Connectivity
        connectivity_stats = self.analyze_connectivity()
        
        # Sparsity
        sparsity_stats = self.sparsity_analysis()
        
        # Data quality
        quality_stats = self.data_quality_assessment()
        
        # Preprocessing
        preprocessed_data, mappings = self.preprocess_graph()
        
        # Save preprocessed graph
        torch.save(preprocessed_data, "preprocessed_poi_graph.pt")
        print(f"\nPreprocessed graph saved to preprocessed_poi_graph.pt")
        
        # Summary report
        self.generate_summary_report(
            edge_stats, degree_stats, feature_stats, 
            edge_attr_stats, shared_stats, connectivity_stats, 
            sparsity_stats, quality_stats
        )
        
        return {
            'preprocessed_data': preprocessed_data,
            'mappings': mappings,
            'analysis_results': {
                'edge_stats': edge_stats,
                'degree_stats': degree_stats,
                'feature_stats': feature_stats,
                'edge_attr_stats': edge_attr_stats,
                'shared_stats': shared_stats,
                'connectivity_stats': connectivity_stats,
                'sparsity_stats': sparsity_stats,
                'quality_stats': quality_stats
            }
        }
    
    def generate_summary_report(self, edge_stats, degree_stats, feature_stats, 
                               edge_attr_stats, shared_stats, connectivity_stats, 
                               sparsity_stats, quality_stats):
        """Generate a comprehensive summary report"""
        print("\n" + "="*60)
        print("COMPREHENSIVE ANALYSIS SUMMARY")
        print("="*60)
        
        print(f"\nGRAPH STRUCTURE:")
        print(f"   - Users: {self.graph['user'].num_nodes}")
        print(f"   - POIs: {self.graph['poi'].num_nodes}")
        print(f"   - SensoryAttributes (Shared): {self.graph['sensory_attr'].num_nodes}")
        print(f"   - OtherAttributes (Shared, VADER): {self.graph['other_attr'].num_nodes}")
        print(f"   - Categories: {self.graph['category'].num_nodes}")
        print(f"   - Total edges: {sum(edge_stats.values())}")
        
        print(f"\nSHARED NODE STATISTICS:")
        if 'sensory_attr' in shared_stats:
            sens_stats = shared_stats['sensory_attr']
            print(f"   - Sensory nodes total: {sens_stats['total_nodes']}")
            print(f"   - Shared between users & POIs: {sens_stats['shared']} ({sens_stats['sharing_ratio']*100:.1f}%)")
        if 'other_attr' in shared_stats:
            print(f"   - Other attribute nodes (with polarity): {shared_stats['other_attr']['total_nodes']}")
        
        print(f"\nDATA QUALITY:")
        print(f"   - Sentiment coverage: {quality_stats['coverage']['sentiment']:.1f}% of POIs")
        print(f"   - Sensory coverage: {quality_stats['coverage']['sensory']:.1f}% of POIs")
        print(f"   - Other attribute coverage (VADER): {quality_stats['coverage']['other']:.1f}% of POIs")
        print(f"   - Reviews with analysis: {quality_stats['processing_stats'].get('total_reviews_processed', 'N/A')}")
        
        print(f"\nCONNECTIVITY:")
        for etype, stats in connectivity_stats.items():
            print(f"   - {etype[1]}: {stats['connectivity_ratio']:.1%} connectivity")
        
        print(f"\nSPARSITY:")
        for etype, stats in sparsity_stats.items():
            print(f"   - {etype[1]}: {stats['density']:.6f} density ({stats['sparsity']:.2%} sparse)")
        
        if edge_attr_stats:
            print(f"\nEDGE ATTRIBUTES:")
            if 'ratings' in edge_attr_stats:
                rating_stats = edge_attr_stats['ratings']
                print(f"   - Ratings: μ={rating_stats['mean']:.2f}, σ={rating_stats['std']:.2f}")
            if 'preferences' in edge_attr_stats:
                pref_stats = edge_attr_stats['preferences']
                print(f"   - Preferences: μ={pref_stats['mean']:.2f}, σ={pref_stats['std']:.2f}")
            if 'other_attr_sentiments' in edge_attr_stats:
                sent_stats = edge_attr_stats['other_attr_sentiments']
                print(f"   - VADER Sentiments: μ={sent_stats['mean']:.2f}, σ={sent_stats['std']:.2f}")
                print(f"     Pos: {sent_stats['positive_count']}, Neu: {sent_stats['neutral_count']}, Neg: {sent_stats['negative_count']}")
        
        print(f"\nKEY INSIGHTS:")
        print(f"   ✓ VADER sentiment analysis for aspect-level attributes (unlimited, fast)")
        print(f"   ✓ Shared other attribute nodes with polarity (Negative/Neutral/Positive)")
        print(f"   ✓ Sentiment scores as edge weights enable nuanced GNN learning")
        print(f"   ✓ Shared sensory nodes enable user-POI attribute matching")
        print(f"   ✓ Heterogeneous structure supports multi-relational learning")
        print(f"   ✓ Autism-specific sensory attributes provide unique signals")


def visualize_graph_structure(graph_path, num_nodes=200):
    """Create a network visualization of the graph structure"""
    print("\nCreating graph structure visualization...")
    
    # Load graph
    data_dict = torch.load(graph_path, weights_only=False)
    graph = data_dict['graph']
    
    # Build NetworkX graph with limited nodes for visualization
    G = nx.Graph()
    node_types = {}
    edge_types = {}
    
    # Add nodes with type information
    node_counters = defaultdict(int)
    max_nodes_per_type = num_nodes // 5
    
    for ntype in graph.node_types:
        for i in range(min(max_nodes_per_type, graph[ntype].num_nodes)):
            node_id = f"{ntype}_{i}"
            G.add_node(node_id)
            node_types[node_id] = ntype
            node_counters[ntype] += 1
    
    # Add edges (limited)
    edge_limit = 2000
    for etype in graph.edge_types:
        if 'rev_' in etype[1]:
            continue
            
        edge_index = graph[etype].edge_index
        src_type, rel_type, dst_type = etype
        
        edges_added = 0
        max_edges_per_type = edge_limit // len([e for e in graph.edge_types if 'rev_' not in e[1]])
        
        for i in range(min(edge_limit, edge_index.size(1))):
            if edges_added >= max_edges_per_type:
                break
                
            src_id = edge_index[0, i].item()
            dst_id = edge_index[1, i].item()
            
            src_node = f"{src_type}_{src_id}"
            dst_node = f"{dst_type}_{dst_id}"
            
            if src_node in G.nodes() and dst_node in G.nodes():
                G.add_edge(src_node, dst_node)
                edge_types[(src_node, dst_node)] = rel_type
                edges_added += 1
    
    # Create visualization
    plt.figure(figsize=(16, 14))
    
    # Use spring layout
    pos = nx.spring_layout(G, k=1.5, iterations=50, seed=42)
    
    # Color mapping for node types
    color_map = {
        'user': '#FF6B6B', 
        'poi': '#4ECDC4', 
        'sensory_attr': '#95E1D3',
        'other_attr': '#F38181',
        'category': '#F6BB4F'
    }
    node_colors = [color_map[node_types[node]] for node in G.nodes()]
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, 
                          node_size=100, alpha=0.8, linewidths=0.5, edgecolors='black')
    
    # Draw edges with different styles
    edge_styles = {
        'rates': '-', 
        'prefers': '--', 
        'belongs_to': '-.', 
        'visits': ':',
        'has_sensory_preference': '-',
        'has_sensory_attribute': '-',
        'has_other_attribute': '--'
    }
    edge_colors = {
        'rates': '#FF9999', 
        'prefers': '#99FF99', 
        'belongs_to': '#9999FF', 
        'visits': '#93DDFF',
        'has_sensory_preference': '#B8E6B8',
        'has_sensory_attribute': '#FFB3BA',
        'has_other_attribute': '#FFDFBA'
    }
    
    for rel_type in set(edge_types.values()):
        edges_of_type = [(u, v) for (u, v), et in edge_types.items() if et == rel_type]
        if edges_of_type:
            nx.draw_networkx_edges(G, pos, edgelist=edges_of_type,
                                 edge_color=edge_colors.get(rel_type, 'gray'),
                                 style=edge_styles.get(rel_type, '-'),
                                 alpha=0.5, width=1.0)
    
    # Create legend
    legend_elements = []
    for ntype, color in color_map.items():
        label = ntype.replace('_', ' ').title()
        legend_elements.append(plt.scatter([], [], c=color, s=100, label=label))
    
    for rel_type, color in edge_colors.items():
        label = rel_type.replace('_', ' ').title()
        legend_elements.append(plt.plot([], [], color=color, 
                                      linestyle=edge_styles.get(rel_type, '-'),
                                      linewidth=2, label=label)[0])
    
    plt.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.2, 1), fontsize=9)
    plt.title('POI Recommendation Graph with VADER Aspect Sentiment\n(Subset visualization with shared attribute nodes)')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('graph_structure_visualization.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Visualized {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    print("Graph structure visualization saved as 'graph_structure_visualization.png'")


# Usage example
if __name__ == "__main__":
    # Initialize analyzer
    analyzer = POIGraphAnalyzer("poi_graph_vader_aspects.pt")
    
    # Run comprehensive analysis
    results = analyzer.comprehensive_analysis()
    
    # Create graph structure visualization
    visualize_graph_structure("poi_graph_vader_aspects.pt")
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    print("\nGenerated files:")
    print("  - preprocessed_poi_graph.pt (preprocessed graph)")
    print("  - poi_preprocessing_mappings.pkl (preprocessing mappings)")
    print("  - degree_distributions.png")
    print("  - user_features_analysis.png")
    print("  - poi_features_analysis.png")
    print("  - sensory_attr_features_analysis.png")
    print("  - other_attr_features_analysis.png (NEW: with VADER polarity)")
    print("  - category_features_analysis.png")
    print("  - rating_distribution.png")
    print("  - preference_distribution.png")
    print("  - other_attr_sentiment_distribution.png (NEW: VADER edge weights)")
    print("  - sensory_node_sharing.png")
    print("  - other_attr_connections.png (NEW: stacked by polarity)")
    print("  - graph_structure_visualization.png")
    
    print("\nNext steps:")
    print("  1. Use preprocessed_poi_graph.pt for model training")
    print("  2. Leverage VADER sentiment scores in GNN message passing")
    print("  3. Implement heterogeneous GNN models (e.g., HGT, HAN)")
    print("  4. Evaluate cold-start performance")
    print("  5. Analyze aspect sentiment impact on recommendations")
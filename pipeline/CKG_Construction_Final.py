import torch
import torch.nn.functional as F
from torch_geometric.data import HeteroData
import numpy as np
from pymongo import MongoClient
from collections import defaultdict
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

class POIGraphBuilder:
    def __init__(self, mongo_uri="mongodb://localhost:27017/", db_name="POIRS"):
        """
        Initialize the graph builder with MongoDB connection
        """
        self.client = MongoClient(mongo_uri)
        self.db = self.client[db_name]
        
        # Collections
        self.reviews_collection = self.db.reviews
        self.images_collection = self.db.images
        self.pois_collection = self.db.pois
        self.questionnaires_collection = self.db.questionnaires
        
        # Node mappings
        self.user_to_idx = {}
        self.poi_to_idx = {}
        self.category_to_idx = {}
        self.sensory_attr_to_idx = {}
        self.other_attr_to_idx = {}
        
        # Sensory attribute types
        self.sensory_types = ['crowd', 'noise', 'space', 'brightness', 'visual_clutter']
        
        # Other attribute types (7 shared nodes)
        self.other_attr_types = ['food', 'staff', 'service', 'ambiance', 'cleanliness', 'atmosphere', 'facilities']
        
        # User attribute mapping
        self.user_attribute_mapping = {
            'cramped_space': ('space', True),         # Invert
            'bright_lighting': ('brightness', False),  # Direct
            'dim_lighting': ('brightness', True),      # Invert
            'noise': ('noise', False),                 # Direct
            'crowd': ('crowd', False),                 # Direct
            'clutter': ('visual_clutter', False)       # Direct
        }

        # Matching strategies
        self.matching_strategies = {
            'noise': 'less_than_equal',
            'crowd': 'less_than_equal',
            'visual_clutter': 'less_than_equal',
            'space': 'greater_than_equal',
            'brightness': 'bidirectional'
        }
        
        # Initialize VADER sentiment analyzer
        self.vader_analyzer = SentimentIntensityAnalyzer()
        print("✅ VADER sentiment analyzer initialized (unlimited, blazingly fast!)")
        
    def analyze_aspect_sentiment_vader(self, aspect, relation, description):
        """
        Analyze sentiment for aspect triple using VADER
        
        Args:
            aspect: The aspect (e.g., "Staff", "Food")
            relation: The relation (e.g., "are", "was")
            description: The description (e.g., "extremely warm and welcoming")
            
        Returns:
            float: Sentiment score from -1 to 1
        """
        # VADER works best with the full statement
        text = f"{aspect} {relation} {description}"
        
        # Get compound score (already normalized to -1 to 1)
        scores = self.vader_analyzer.polarity_scores(text)
        return scores['compound']
        
    def load_data(self):
        """Load and preprocess data from MongoDB collections"""
        print("Loading data from MongoDB...")
        
        # Load all collections
        self.users_data = list(self.questionnaires_collection.find())
        self.pois_data = list(self.pois_collection.find())
        
        # FILTER: Only load reviews with non-null analysis attributes
        print("Filtering reviews with valid analysis...")
        
        total_reviews = self.reviews_collection.count_documents({})
        
        reviews_query = {
            "analysis": {
                "$ne": None,
                "$exists": True
            },
            "analysis.error": {
                "$exists": False
            }
        }
        
        self.reviews_data = list(self.reviews_collection.find(reviews_query))
        self.images_data = list(self.images_collection.find())
        
        filtered_count = len(self.reviews_data)
        filter_rate = (filtered_count / total_reviews) * 100 if total_reviews > 0 else 0
        
        print(f"Review filtering results:")
        print(f"  Total reviews in collection: {total_reviews}")
        print(f"  Reviews with valid analysis: {filtered_count}")
        print(f"  Filter retention rate: {filter_rate:.1f}%")
        
        print(f"Loaded {len(self.users_data)} users, {len(self.pois_data)} POIs")
        print(f"Loaded {filtered_count} analyzed reviews, {len(self.images_data)} images")
        
        self._validate_analysis_content()
        
    def _validate_analysis_content(self):
        """Validate the quality of analysis content in filtered reviews"""
        print("Validating analysis content quality...")
        
        valid_sentiment = 0
        valid_triples = 0
        valid_sensory = 0
        total_analyzed = len(self.reviews_data)
        
        sensory_attr_counts = defaultdict(int)
        triple_counts = []
        
        for review in self.reviews_data:
            analysis = review.get('analysis', {})
            
            if 'sentiment' in analysis and isinstance(analysis['sentiment'], (int, float)):
                valid_sentiment += 1
            
            triples = analysis.get('triples', [])
            if isinstance(triples, list) and len(triples) > 0:
                valid_triples += 1
                triple_counts.append(len(triples))
            
            sensory = analysis.get('sensory', {})
            if isinstance(sensory, dict):
                sensory_scores = {k: v for k, v in sensory.items() if not k.endswith('_reason') and v is not None}
                if sensory_scores:
                    valid_sensory += 1
                    for attr_type in sensory_scores.keys():
                        if attr_type in self.sensory_types:
                            sensory_attr_counts[attr_type] += 1
        
        print(f"Analysis content validation:")
        print(f"  Reviews with valid sentiment: {valid_sentiment}/{total_analyzed} ({valid_sentiment/total_analyzed*100:.1f}%)")
        print(f"  Reviews with valid triples: {valid_triples}/{total_analyzed} ({valid_triples/total_analyzed*100:.1f}%)")
        print(f"  Reviews with valid sensory: {valid_sensory}/{total_analyzed} ({valid_sensory/total_analyzed*100:.1f}%)")
        
        if triple_counts:
            print(f"  Average triples per review: {np.mean(triple_counts):.2f}")
            print(f"  Max triples in a review: {max(triple_counts)}")
        
        print(f"  Sensory attribute coverage:")
        for attr_type in self.sensory_types:
            count = sensory_attr_counts[attr_type]
            coverage = (count / total_analyzed) * 100 if total_analyzed > 0 else 0
            print(f"    {attr_type}: {count} reviews ({coverage:.1f}%)")
    
    def create_node_mappings(self):
        """Create mappings from IDs to sequential indices"""
        # User mappings
        for i, user in enumerate(self.users_data):
            self.user_to_idx[user['userId']] = i
            
        # POI mappings  
        for i, poi in enumerate(self.pois_data):
            self.poi_to_idx[poi['poi_id']] = i
            
        # Category mappings
        categories = list(set([poi['category'] for poi in self.pois_data]))
        for i, category in enumerate(categories):
            self.category_to_idx[category] = i
        
        print(f"Created mappings: {len(self.user_to_idx)} users, {len(self.poi_to_idx)} POIs, {len(self.category_to_idx)} categories")
    
    def consolidate_poi_sentiment(self):
        """
        Consolidate sentiment scores for each POI using Simple Majority Polarity Rule
        """
        poi_sentiment_data = defaultdict(list)
        
        reviews_with_sentiment = 0
        
        for review in self.reviews_data:
            poi_id = review['poi_id']
            if poi_id in self.poi_to_idx:
                analysis = review.get('analysis', {})
                sentiment = analysis.get('sentiment')
                
                if sentiment is not None:
                    try:
                        float_val = float(sentiment)
                        if 0 <= float_val <= 1:
                            poi_sentiment_data[poi_id].append(float_val)
                            reviews_with_sentiment += 1
                    except (ValueError, TypeError):
                        continue
        
        print(f"Sentiment consolidation statistics:")
        print(f"  Reviews with valid sentiment: {reviews_with_sentiment}")
        
        # Apply Simple Majority Polarity Rule
        self.poi_consolidated_sentiment = {}
        
        for poi_id, values in poi_sentiment_data.items():
            if not values:
                continue
            
            # Bucket values
            low_bucket = [v for v in values if v <= 0.2]
            moderate_bucket = [v for v in values if 0.2 < v < 0.4]
            high_bucket = [v for v in values if v >= 0.4]
            
            buckets = [
                ('Low', low_bucket),
                ('Moderate', moderate_bucket), 
                ('High', high_bucket)
            ]
            
            majority_bucket = max(buckets, key=lambda x: len(x[1]))
            polarity_direction = majority_bucket[0]
            majority_values = majority_bucket[1]
            
            if majority_values:
                consolidated_value = np.mean(majority_values)
                self.poi_consolidated_sentiment[poi_id] = {
                    'consolidated_value': consolidated_value,
                    'polarity_direction': polarity_direction,
                    'source_count': len(values)
                }
        
        print(f"Consolidated sentiment for {len(self.poi_consolidated_sentiment)} POIs")
        
        # Statistics
        sentiment_dist = defaultdict(int)
        for data in self.poi_consolidated_sentiment.values():
            sentiment_dist[data['polarity_direction']] += 1
        
        print(f"Sentiment polarity distribution:")
        for polarity, count in sentiment_dist.items():
            print(f"  {polarity}: {count} POIs")
    
    def consolidate_poi_sensory_attributes(self):
        """
        Consolidate sensory attributes for each POI using Simple Majority Polarity Rule
        """
        poi_sensory_data = defaultdict(lambda: defaultdict(list))
        
        reviews_with_sensory = 0
        total_sensory_extractions = 0
        
        for review in self.reviews_data:
            poi_id = review['poi_id']
            if poi_id in self.poi_to_idx:
                analysis = review.get('analysis', {})
                sensory = analysis.get('sensory', {})
                
                if isinstance(sensory, dict):
                    has_sensory_data = False
                    for attr_type, value in sensory.items():
                        if not attr_type.endswith('_reason') and value is not None:
                            try:
                                float_val = float(value)
                                if 1 <= float_val <= 5:
                                    poi_sensory_data[poi_id][attr_type].append(float_val)
                                    has_sensory_data = True
                                    total_sensory_extractions += 1
                            except (ValueError, TypeError):
                                continue
                    
                    if has_sensory_data:
                        reviews_with_sensory += 1
        
        # Collect from images
        images_with_sensory = 0
        for image in self.images_data:
            poi_id = image['poi_id']
            if poi_id in self.poi_to_idx and 'details' in image:
                has_image_sensory = False
                for attr_type, details in image['details'].items():
                    if isinstance(details, dict) and 'scale' in details:
                        try:
                            scale_val = float(details['scale'])
                            if 1 <= scale_val <= 5:
                                poi_sensory_data[poi_id][attr_type].append(scale_val)
                                has_image_sensory = True
                        except (ValueError, TypeError):
                            continue
                
                if has_image_sensory:
                    images_with_sensory += 1
        
        print(f"Sensory data collection statistics:")
        print(f"  Reviews with sensory data: {reviews_with_sensory}")
        print(f"  Images with sensory data: {images_with_sensory}")
        print(f"  Total sensory value extractions: {total_sensory_extractions}")
        
        # Apply Simple Majority Polarity Rule
        self.poi_consolidated_sensory = {}
        
        for poi_id, attributes in poi_sensory_data.items():
            self.poi_consolidated_sensory[poi_id] = {}
            
            for attr_type, values in attributes.items():
                if not values:
                    continue
                
                low_bucket = [v for v in values if v <= 2.0]
                moderate_bucket = [v for v in values if 2.0 < v < 4.0]
                high_bucket = [v for v in values if v >= 4.0]
                
                buckets = [
                    ('Low', low_bucket),
                    ('Moderate', moderate_bucket), 
                    ('High', high_bucket)
                ]
                
                majority_bucket = max(buckets, key=lambda x: len(x[1]))
                polarity_direction = majority_bucket[0]
                majority_values = majority_bucket[1]
                
                if majority_values:
                    consolidated_value = np.mean(majority_values)
                    # Round to 1 decimal place for shared nodes
                    consolidated_value = round(consolidated_value, 1)
                    
                    self.poi_consolidated_sensory[poi_id][attr_type] = {
                        'consolidated_value': consolidated_value,
                        'polarity_direction': polarity_direction,
                        'source_count': len(values)
                    }
        
        print(f"Consolidated sensory attributes for {len(self.poi_consolidated_sensory)} POIs")
    
    def extract_poi_other_attributes_with_vader(self):
        """
        Extract aspect-level sentiment using VADER (unlimited, fast!)
        Consolidates sentiment per aspect type for each POI
        """
        print("Starting VADER aspect sentiment extraction...")
        
        aspect_sentiments = defaultdict(lambda: defaultdict(list))
        reviews_with_triples = 0
        total_triples_processed = 0
        
        import time
        start_time = time.time()
        
        for review in self.reviews_data:
            poi_id = review['poi_id']
            
            if poi_id not in self.poi_to_idx:
                continue
            
            analysis = review.get('analysis', {})
            triples = analysis.get('triples', [])
            
            if isinstance(triples, list) and len(triples) > 0:
                reviews_with_triples += 1
                
                for triple in triples:
                    if isinstance(triple, (list, tuple)) and len(triple) == 3:
                        try:
                            aspect, relation, description = triple
                            if not all(isinstance(x, str) and len(x.strip()) > 0 
                                      for x in [aspect, relation, description]):
                                continue
                            
                            aspect_key = aspect.lower().strip()
                            
                            # Only process predefined aspect types
                            if aspect_key not in self.other_attr_types:
                                continue
                            
                            # Analyze sentiment with VADER
                            sentiment_score = self.analyze_aspect_sentiment_vader(
                                aspect, relation, description
                            )
                            
                            aspect_sentiments[poi_id][aspect_key].append(sentiment_score)
                            total_triples_processed += 1
                            
                        except Exception as e:
                            continue
        
        elapsed_time = time.time() - start_time
        print(f"\n✅ VADER sentiment analysis completed in {elapsed_time:.2f} seconds")
        print(f"   Processing rate: {total_triples_processed/elapsed_time:.0f} triples/sec")
        
        # Consolidate using majority polarity rule
        self.poi_other_attributes = {}
        
        for poi_id, aspects in aspect_sentiments.items():
            self.poi_other_attributes[poi_id] = {}
            
            for aspect_key, sentiments in aspects.items():
                if not sentiments:
                    continue
                
                # Bucket sentiments
                negative_bucket = [s for s in sentiments if s <= -0.3]
                neutral_bucket = [s for s in sentiments if -0.3 < s < 0.3]
                positive_bucket = [s for s in sentiments if s >= 0.3]
                
                buckets = [
                    ('Negative', negative_bucket),
                    ('Neutral', neutral_bucket), 
                    ('Positive', positive_bucket)
                ]
                
                majority_bucket = max(buckets, key=lambda x: len(x[1]))
                polarity_direction = majority_bucket[0]
                majority_values = majority_bucket[1]
                
                if majority_values:
                    consolidated_sentiment = np.mean(majority_values)
                    
                    self.poi_other_attributes[poi_id][aspect_key] = {
                        'consolidated_sentiment': consolidated_sentiment,
                        'polarity_direction': polarity_direction,
                        'source_count': len(sentiments),
                        'sentiment_std': np.std(sentiments)
                    }
        
        print(f"\n📊 Aspect Sentiment Extraction Statistics:")
        print(f"  Reviews with valid triples: {reviews_with_triples}")
        print(f"  Total relevant triples processed: {total_triples_processed}")
        print(f"  POIs with other attributes: {len(self.poi_other_attributes)}")
        
        # Coverage statistics
        attr_coverage = defaultdict(int)
        sentiment_dist = defaultdict(lambda: defaultdict(int))
        avg_sentiments = defaultdict(list)
        
        for poi_attrs in self.poi_other_attributes.values():
            for aspect_key, data in poi_attrs.items():
                attr_coverage[aspect_key] += 1
                sentiment_dist[aspect_key][data['polarity_direction']] += 1
                avg_sentiments[aspect_key].append(data['consolidated_sentiment'])
        
        print(f"\n  📈 Attribute Coverage:")
        for attr_type in self.other_attr_types:
            count = attr_coverage[attr_type]
            coverage = (count / len(self.poi_to_idx)) * 100 if len(self.poi_to_idx) > 0 else 0
            avg_sent = np.mean(avg_sentiments[attr_type]) if avg_sentiments[attr_type] else 0
            
            print(f"    {attr_type}: {count} POIs ({coverage:.1f}%), avg sentiment: {avg_sent:.2f}")
            if count > 0:
                print(f"      Positive: {sentiment_dist[attr_type]['Positive']}, "
                      f"Neutral: {sentiment_dist[attr_type]['Neutral']}, "
                      f"Negative: {sentiment_dist[attr_type]['Negative']}")
    
    def extract_user_sensory_preferences(self):
        """Extract and process user sensory preferences"""
        self.user_sensory_preferences = {}
        
        for user in self.users_data:
            user_id = user['userId']
            self.user_sensory_preferences[user_id] = {}
            
            for original_attr in ['crowd', 'cramped_space', 'bright_lighting', 'dim_lighting', 'noise', 'clutter']:
                user_comfort = user['sensory'].get(original_attr, 3)
                
                if original_attr in self.user_attribute_mapping:
                    std_attr, invert = self.user_attribute_mapping[original_attr]
                    if invert:
                        mapped_comfort = 6 - user_comfort
                    else:
                        mapped_comfort = user_comfort
                    
                    # Round to 1 decimal place for shared nodes
                    mapped_comfort = round(float(mapped_comfort), 1)
                    self.user_sensory_preferences[user_id][std_attr] = mapped_comfort
                else:
                    self.user_sensory_preferences[user_id][original_attr] = round(float(user_comfort), 1)
        
        print(f"Extracted sensory preferences for {len(self.user_sensory_preferences)} users")
    
    def create_sensory_attribute_nodes(self):
        """Create shared SensoryAttribute nodes for (type, value) combinations"""
        sensory_attr_set = set()
        
        # Collect from POIs
        for poi_id, attributes in self.poi_consolidated_sensory.items():
            for attr_type, data in attributes.items():
                if attr_type in self.sensory_types:
                    value = data['consolidated_value']
                    sensory_attr_set.add((attr_type, value))
        
        # Collect from Users
        for user_id, preferences in self.user_sensory_preferences.items():
            for attr_type, value in preferences.items():
                if attr_type in self.sensory_types:
                    sensory_attr_set.add((attr_type, value))
        
        # Create mappings
        for i, (attr_type, value) in enumerate(sorted(sensory_attr_set)):
            self.sensory_attr_to_idx[(attr_type, value)] = i
        
        print(f"Created {len(self.sensory_attr_to_idx)} shared SensoryAttribute nodes")
        
        # Statistics
        type_counts = defaultdict(int)
        for attr_type, value in sensory_attr_set:
            type_counts[attr_type] += 1
        
        print(f"  SensoryAttribute distribution by type:")
        for attr_type in self.sensory_types:
            print(f"    {attr_type}: {type_counts[attr_type]} unique values")
    
    def create_other_attribute_nodes(self):
        """Create shared OtherAttribute nodes with (type, sentiment_bucket) combinations"""
        other_attr_set = set()
        
        # Collect from POIs with sentiment buckets
        for poi_id, attributes in self.poi_other_attributes.items():
            for attr_type, data in attributes.items():
                if attr_type in self.other_attr_types:
                    polarity = data['polarity_direction']
                    other_attr_set.add((attr_type, polarity))
        
        # Create mappings
        self.other_attr_to_idx = {}
        for i, (attr_type, polarity) in enumerate(sorted(other_attr_set)):
            self.other_attr_to_idx[(attr_type, polarity)] = i
        
        print(f"Created {len(self.other_attr_to_idx)} shared OtherAttribute nodes")
        
        # Statistics
        type_counts = defaultdict(int)
        for attr_type, polarity in other_attr_set:
            type_counts[attr_type] += 1
        
        print(f"  OtherAttribute distribution by type:")
        for attr_type in self.other_attr_types:
            print(f"    {attr_type}: {type_counts[attr_type]} polarity nodes")
    
    def create_node_features(self):
        """Create feature tensors for all node types"""
        
        # User features: age + gender only
        user_features = []
        age_groups = ['18-25', '26-35', '36-45', '46-55', '56+']
        genders = ['Male', 'Female', 'Other']
        
        for user in self.users_data:
            features = []
            
            # Age one-hot
            age_onehot = [1 if user['age'] == ag else 0 for ag in age_groups]
            features.extend(age_onehot)
            
            # Gender one-hot
            gender_onehot = [1 if user['gender'] == g else 0 for g in genders]
            features.extend(gender_onehot)
            
            user_features.append(features)
        
        self.user_features = torch.tensor(user_features, dtype=torch.float)
        
        # POI features: sentiment score only
        poi_features = []
        
        for poi in self.pois_data:
            poi_id = poi['poi_id']
            
            # Get consolidated sentiment, default to 3.0 (neutral) if not available
            if poi_id in self.poi_consolidated_sentiment:
                sentiment = self.poi_consolidated_sentiment[poi_id]['consolidated_value']
            else:
                sentiment = 3.0
            
            poi_features.append([sentiment])
        
        self.poi_features = torch.tensor(poi_features, dtype=torch.float)
        
        # SensoryAttribute features: type (one-hot) + value
        sensory_features = []
        
        for (attr_type, value) in sorted(self.sensory_attr_to_idx.keys()):
            features = []
            
            # Type one-hot
            type_onehot = [1 if attr_type == t else 0 for t in self.sensory_types]
            features.extend(type_onehot)
            
            # Value
            features.append(value)
            
            sensory_features.append(features)
        
        self.sensory_features = torch.tensor(sensory_features, dtype=torch.float)
        
        # OtherAttribute features: type (one-hot) + polarity (one-hot) + sentiment value
        other_features = []
        polarity_types = ['Negative', 'Neutral', 'Positive']
        
        for (attr_type, polarity) in sorted(self.other_attr_to_idx.keys()):
            features = []
            
            # Type one-hot
            type_onehot = [1 if attr_type == t else 0 for t in self.other_attr_types]
            features.extend(type_onehot)
            
            # Polarity one-hot
            polarity_onehot = [1 if polarity == p else 0 for p in polarity_types]
            features.extend(polarity_onehot)
            
            # Sentiment value mapping
            sentiment_value = {'Negative': -0.5, 'Neutral': 0.0, 'Positive': 0.5}[polarity]
            features.append(sentiment_value)
            
            other_features.append(features)
        
        self.other_features = torch.tensor(other_features, dtype=torch.float)
        
        # Category features: one-hot
        category_features = []
        for category in sorted(self.category_to_idx.keys()):
            features = [0] * len(self.category_to_idx)
            features[self.category_to_idx[category]] = 1
            category_features.append(features)
        
        self.category_features = torch.tensor(category_features, dtype=torch.float)
        
        print(f"Created features:")
        print(f"  Users: {self.user_features.shape}")
        print(f"  POIs: {self.poi_features.shape}")
        print(f"  SensoryAttributes: {self.sensory_features.shape}")
        print(f"  OtherAttributes: {self.other_features.shape}")
        print(f"  Categories: {self.category_features.shape}")
    
    def _get_compatible_sensory_nodes(self, user_value, attr_type):
        """
        Get all sensory attribute nodes compatible with user's preference
        
        Args:
            user_value: User's mapped preference value (1-5)
            attr_type: Type of sensory attribute
            
        Returns:
            List of (node_key, compatibility_score) tuples
        """
        compatible_nodes = []
        matching_strategy = self.matching_strategies.get(attr_type, 'exact')
        
        for (node_attr_type, node_value) in self.sensory_attr_to_idx.keys():
            if node_attr_type != attr_type:
                continue
            
            compatibility = 0.0
            
            if matching_strategy == 'less_than_equal':
                # User wants POIs with values ≤ their comfort level
                if node_value <= user_value:
                    compatibility = 1.0  # Perfect match
                else:
                    # Penalize higher values (decay rapidly)
                    compatibility = max(0, 1 - 0.3 * (node_value - user_value))
            
            elif matching_strategy == 'greater_than_equal':
                # User wants POIs with values ≥ their comfort level
                if node_value >= user_value:
                    compatibility = 1.0  # Perfect match
                else:
                    # Penalize lower values
                    compatibility = max(0, 1 - 0.3 * (user_value - node_value))
            
            elif matching_strategy == 'bidirectional':
                # For brightness: exact match is best, with symmetric decay
                diff = abs(node_value - user_value)
                compatibility = max(0, 1 - 0.25 * diff)
            
            else:  # 'exact'
                compatibility = 1.0 if abs(node_value - user_value) < 0.1 else 0.0
            
            # Only include nodes with meaningful compatibility
            if compatibility > 0.05:  # Threshold to avoid very weak connections
                compatible_nodes.append(((node_attr_type, node_value), compatibility))
        
        return compatible_nodes

    def create_edges(self):
        """Create edge indices and attributes for all relationships"""
        
        # User-POI edges (ratings and visits)
        user_poi_edges = []
        user_poi_ratings = []
        
        for user in self.users_data:
            user_idx = self.user_to_idx[user['userId']]
            
            for poi_id, rating in user['poi_ratings'].items():
                if poi_id in self.poi_to_idx and rating != 'Not Visited':
                    poi_idx = self.poi_to_idx[poi_id]
                    user_poi_edges.append([user_idx, poi_idx])
                    user_poi_ratings.append(float(rating))
        
        # User-Category edges
        user_category_edges = []
        user_category_preferences = []
        
        for user in self.users_data:
            user_idx = self.user_to_idx[user['userId']]
            
            for category, preference in user['categories'].items():
                if category in self.category_to_idx:
                    category_idx = self.category_to_idx[category]
                    user_category_edges.append([user_idx, category_idx])
                    user_category_preferences.append(float(preference))
        
        # POI-Category edges
        poi_category_edges = []
        
        for poi in self.pois_data:
            poi_idx = self.poi_to_idx[poi['poi_id']]
            category_idx = self.category_to_idx[poi['category']]
            poi_category_edges.append([poi_idx, category_idx])
        
        # User-SensoryAttribute edges with multiple connections
        user_sensory_edges = []
        user_sensory_compatibility = []
        
        total_user_sensory_connections = 0
        users_with_multiple_edges = 0
        
        for user in self.users_data:
            user_idx = self.user_to_idx[user['userId']]
            user_id = user['userId']
            
            if user_id in self.user_sensory_preferences:
                user_edge_count = 0
                for attr_type, user_value in self.user_sensory_preferences[user_id].items():
                    # Get all compatible sensory nodes for this attribute
                    compatible_nodes = self._get_compatible_sensory_nodes(user_value, attr_type)
                    
                    for (node_key, compatibility) in compatible_nodes:
                        if node_key in self.sensory_attr_to_idx:
                            sensory_idx = self.sensory_attr_to_idx[node_key]
                            user_sensory_edges.append([user_idx, sensory_idx])
                            user_sensory_compatibility.append(compatibility)
                            user_edge_count += 1
                
                total_user_sensory_connections += user_edge_count
                if user_edge_count > len(self.user_sensory_preferences[user_id]):
                    users_with_multiple_edges += 1
        
        print(f"User-Sensory edge statistics:")
        print(f"  Total user-sensory edges: {len(user_sensory_edges)}")
        print(f"  Avg edges per user: {total_user_sensory_connections / len(self.users_data):.2f}")
        print(f"  Users with multiple edges per attribute: {users_with_multiple_edges}")
        
        # POI-SensoryAttribute edges (exact match only for POIs)
        poi_sensory_edges = []
        
        for poi in self.pois_data:
            poi_idx = self.poi_to_idx[poi['poi_id']]
            poi_id = poi['poi_id']
            
            if poi_id in self.poi_consolidated_sensory:
                for attr_type, data in self.poi_consolidated_sensory[poi_id].items():
                    value = data['consolidated_value']
                    if (attr_type, value) in self.sensory_attr_to_idx:
                        sensory_idx = self.sensory_attr_to_idx[(attr_type, value)]
                        poi_sensory_edges.append([poi_idx, sensory_idx])
        
        # POI-OtherAttribute edges with sentiment weights
        poi_other_edges = []
        poi_other_sentiments = []
        
        for poi in self.pois_data:
            poi_idx = self.poi_to_idx[poi['poi_id']]
            poi_id = poi['poi_id']
            
            if poi_id in self.poi_other_attributes:
                for attr_type, data in self.poi_other_attributes[poi_id].items():
                    polarity = data['polarity_direction']
                    sentiment = data['consolidated_sentiment']
                    
                    if (attr_type, polarity) in self.other_attr_to_idx:
                        other_idx = self.other_attr_to_idx[(attr_type, polarity)]
                        poi_other_edges.append([poi_idx, other_idx])
                        poi_other_sentiments.append(sentiment)
        
        # Convert to tensors
        self.edge_indices = {
            ('user', 'rates', 'poi'): torch.tensor(user_poi_edges, dtype=torch.long).t().contiguous() if user_poi_edges else torch.empty((2, 0), dtype=torch.long),
            ('user', 'visits', 'poi'): torch.tensor(user_poi_edges, dtype=torch.long).t().contiguous() if user_poi_edges else torch.empty((2, 0), dtype=torch.long),
            ('user', 'prefers', 'category'): torch.tensor(user_category_edges, dtype=torch.long).t().contiguous() if user_category_edges else torch.empty((2, 0), dtype=torch.long),
            ('poi', 'belongs_to', 'category'): torch.tensor(poi_category_edges, dtype=torch.long).t().contiguous() if poi_category_edges else torch.empty((2, 0), dtype=torch.long),
            ('user', 'has_sensory_preference', 'sensory_attr'): torch.tensor(user_sensory_edges, dtype=torch.long).t().contiguous() if user_sensory_edges else torch.empty((2, 0), dtype=torch.long),
            ('poi', 'has_sensory_attribute', 'sensory_attr'): torch.tensor(poi_sensory_edges, dtype=torch.long).t().contiguous() if poi_sensory_edges else torch.empty((2, 0), dtype=torch.long),
            ('poi', 'has_other_attribute', 'other_attr'): torch.tensor(poi_other_edges, dtype=torch.long).t().contiguous() if poi_other_edges else torch.empty((2, 0), dtype=torch.long),
        }
        
        self.edge_attributes = {
            ('user', 'rates', 'poi'): torch.tensor(user_poi_ratings, dtype=torch.float).unsqueeze(1) if user_poi_ratings else torch.empty((0, 1), dtype=torch.float),
            ('user', 'prefers', 'category'): torch.tensor(user_category_preferences, dtype=torch.float).unsqueeze(1) if user_category_preferences else torch.empty((0, 1), dtype=torch.float),
            ('user', 'has_sensory_preference', 'sensory_attr'): torch.tensor(user_sensory_compatibility, dtype=torch.float).unsqueeze(1) if user_sensory_compatibility else torch.empty((0, 1), dtype=torch.float),
            ('poi', 'has_other_attribute', 'other_attr'): torch.tensor(poi_other_sentiments, dtype=torch.float).unsqueeze(1) if poi_other_sentiments else torch.empty((0, 1), dtype=torch.float),
        }
        
        print("Created edge indices and attributes:")
        for edge_type, edge_idx in self.edge_indices.items():
            print(f"  {edge_type}: {edge_idx.shape}") 
               
    def build_hetero_graph(self):
        """Build the final heterogeneous graph"""
        
        data = HeteroData()
        
        # Add node features
        data['user'].x = self.user_features
        data['poi'].x = self.poi_features
        data['sensory_attr'].x = self.sensory_features
        data['other_attr'].x = self.other_features
        data['category'].x = self.category_features
        
        # Add edges
        for edge_type, edge_index in self.edge_indices.items():
            data[edge_type].edge_index = edge_index
            
            if edge_type in self.edge_attributes:
                data[edge_type].edge_attr = self.edge_attributes[edge_type]
        
        # Add reverse edges
        data[('poi', 'rev_rates', 'user')].edge_index = data[('user', 'rates', 'poi')].edge_index.flip([0])
        data[('poi', 'rev_visits', 'user')].edge_index = data[('user', 'visits', 'poi')].edge_index.flip([0])
        data[('category', 'rev_prefers', 'user')].edge_index = data[('user', 'prefers', 'category')].edge_index.flip([0])
        data[('category', 'rev_belongs_to', 'poi')].edge_index = data[('poi', 'belongs_to', 'category')].edge_index.flip([0])
        data[('sensory_attr', 'rev_has_sensory_preference', 'user')].edge_index = data[('user', 'has_sensory_preference', 'sensory_attr')].edge_index.flip([0])
        data[('sensory_attr', 'rev_has_sensory_attribute', 'poi')].edge_index = data[('poi', 'has_sensory_attribute', 'sensory_attr')].edge_index.flip([0])
        data[('other_attr', 'rev_has_other_attribute', 'poi')].edge_index = data[('poi', 'has_other_attribute', 'other_attr')].edge_index.flip([0])
        
        # Copy edge attributes for reverse edges
        if ('user', 'rates', 'poi') in self.edge_attributes:
            data[('poi', 'rev_rates', 'user')].edge_attr = self.edge_attributes[('user', 'rates', 'poi')]
        if ('user', 'prefers', 'category') in self.edge_attributes:
            data[('category', 'rev_prefers', 'user')].edge_attr = self.edge_attributes[('user', 'prefers', 'category')]
        if ('user', 'has_sensory_preference', 'sensory_attr') in self.edge_attributes:
            data[('sensory_attr', 'rev_has_sensory_preference', 'user')].edge_attr = self.edge_attributes[('user', 'has_sensory_preference', 'sensory_attr')]
        if ('poi', 'has_other_attribute', 'other_attr') in self.edge_attributes:
            data[('other_attr', 'rev_has_other_attribute', 'poi')].edge_attr = self.edge_attributes[('poi', 'has_other_attribute', 'other_attr')]
        
        self.hetero_data = data
        
        print("\nBuilt heterogeneous graph:")
        print(f"  User nodes: {data['user'].x.shape[0]}")
        print(f"  POI nodes: {data['poi'].x.shape[0]}")
        print(f"  SensoryAttribute nodes: {data['sensory_attr'].x.shape[0]}")
        print(f"  OtherAttribute nodes: {data['other_attr'].x.shape[0]}")
        print(f"  Category nodes: {data['category'].x.shape[0]}")
        
        print("\nFinal data quality summary:")
        print(f"  Reviews used for analysis: {len(self.reviews_data)}")
        print(f"  POIs with sentiment scores: {len(self.poi_consolidated_sentiment)}")
        print(f"  POIs with sensory attributes: {len(self.poi_consolidated_sensory)}")
        print(f"  POIs with other attributes: {len(self.poi_other_attributes)}")
        print(f"  Shared sensory attribute nodes: {len(self.sensory_attr_to_idx)}")
        print(f"  Shared other attribute nodes: {len(self.other_attr_to_idx)}")
        
        return data
    
    def build_graph(self):
        """Main method to build the complete graph with VADER aspect sentiment"""
        print("Building POI Recommendation Graph with VADER Aspect Sentiment...")
        
        # Load data (with filtering)
        self.load_data()
        
        # Create mappings
        self.create_node_mappings()
        
        # Process POI attributes
        self.consolidate_poi_sentiment()
        self.consolidate_poi_sensory_attributes()
        self.extract_poi_other_attributes_with_vader()
        
        # Process user sensory preferences
        self.extract_user_sensory_preferences()
        
        # Create shared sensory attribute nodes
        self.create_sensory_attribute_nodes()
        self.create_other_attribute_nodes()
        
        # Create features and edges
        self.create_node_features()
        self.create_edges()
        
        # Build final graph
        graph = self.build_hetero_graph()
        
        print("\n✅ Graph building completed with VADER aspect sentiment!")
        
        return graph, {
            'user_to_idx': self.user_to_idx,
            'poi_to_idx': self.poi_to_idx,
            'category_to_idx': self.category_to_idx,
            'sensory_attr_to_idx': self.sensory_attr_to_idx,
            'other_attr_to_idx': self.other_attr_to_idx,
            'poi_consolidated_sentiment': self.poi_consolidated_sentiment,
            'poi_consolidated_sensory': self.poi_consolidated_sensory,
            'poi_other_attributes': self.poi_other_attributes,
            'user_sensory_preferences': self.user_sensory_preferences,
            'statistics': {
                'total_reviews_processed': len(self.reviews_data),
                'pois_with_sentiment': len(self.poi_consolidated_sentiment),
                'pois_with_sensory': len(self.poi_consolidated_sensory),
                'pois_with_other_attrs': len(self.poi_other_attributes),
                'shared_sensory_nodes': len(self.sensory_attr_to_idx),
                'shared_other_nodes': len(self.other_attr_to_idx)
            }
        }

# Usage example
if __name__ == "__main__":
    # Initialize builder
    builder = POIGraphBuilder(
        mongo_uri="mongodb://localhost:27017/",
        db_name="POIRS"
    )
    
    # Build the graph
    graph, metadata = builder.build_graph()
    
    # Print graph statistics
    print("\n" + "="*60)
    print("FINAL GRAPH STATISTICS")
    print("="*60)
    print(graph)
    
    print("\n" + "="*60)
    print("NODE COUNTS")
    print("="*60)
    print(f"Users: {graph['user'].x.shape[0]}")
    print(f"POIs: {graph['poi'].x.shape[0]}")
    print(f"SensoryAttribute (Shared): {graph['sensory_attr'].x.shape[0]}")
    print(f"OtherAttribute (Shared): {graph['other_attr'].x.shape[0]}")
    print(f"Categories: {graph['category'].x.shape[0]}")
    
    print("\n" + "="*60)
    print("EDGE COUNTS")
    print("="*60)
    for edge_type in graph.edge_types:
        edge_count = graph[edge_type].edge_index.shape[1]
        print(f"{edge_type}: {edge_count} edges")
    
    print("\n" + "="*60)
    print("SHARED NODE STATISTICS")
    print("="*60)
    print(f"Unique (type, value) sensory combinations: {metadata['statistics']['shared_sensory_nodes']}")
    print(f"Other attribute (type, polarity) combinations: {metadata['statistics']['shared_other_nodes']}")
    
    # Save the graph and metadata
    torch.save({
        'graph': graph,
        'metadata': metadata
    }, 'poi_graph_vader_aspects.pt')
    
    print("\n" + "="*60)
    print("Graph saved to 'poi_graph_vader_aspects.pt'")
    print("="*60)
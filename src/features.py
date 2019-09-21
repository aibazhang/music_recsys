context_features_time = ['tweet_lang', 'time_zone', 'dayofweek', 'dayofyear', 'hour', 'is_week_day', 'period_of_day']

context_features = ['tweet_lang', 'lang', 'time_zone', 'dayofweek', 'dayofyear', 'hour', 'is_week_day', 'period_of_day']


music_content_features = ['instrumentalness', 'liveness', 'speechiness', 'danceability',
                          'valence', 'loudness', 'tempo', 'acousticness', 'mode', 'key', 'energy', 'genre', 'played_counts']

UGP_features = [
    'UGP_electronic', 'UGP_rock', 'UGP_new age', 'UGP_classical', 'UGP_reggae', 'UGP_blues', 'UGP_country', 'UGP_world', 
    'UGP_folk', 'UGP_easy listening', 'UGP_jazz', 'UGP_vocal', "UGP_children's", 'UGP_punk', 'UGP_alternative', 
    'UGP_spoken word', 'UGP_pop', 'UGP_heavy metal'
]

artist_genre_features = [
    'electronic', 'rock', 'new age', 'classical', 'reggae', 'blues', 'g_country', 'world', 'folk', 
    'easy listening', 'jazz', 'vocal', "children's", 'punk', 'alternative', 'spoken word', 'pop', 'heavy metal'
]

non_categorical_features = music_content_features + UGP_features + artist_genre_features + ['play_counts']
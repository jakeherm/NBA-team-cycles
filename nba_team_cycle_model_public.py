import pandas as pd
import numpy as np
import tensorflow as tf

def create_train_sequences(df, sequence_length=10):
    lstm_data = []
    for team in df['abbreviation'].unique():
        team_data = df[df['abbreviation'] == team].sort_values(by='Season')
        for i in range(len(team_data) - sequence_length - 6):  # Predict 7 years ahead
            seq = team_data.iloc[i:i + sequence_length]['WinPCT'].values
            target = team_data.iloc[i + sequence_length:i + sequence_length + 7]['WinPCT'].values
            lstm_data.append((seq, target))
    return lstm_data

def build_lstm_model(X_train, y_train):
    # Build LSTM model with Dropout for Monte Carlo Dropout
    baseline_model = tf.keras.Sequential([
        tf.keras.layers.LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
        tf.keras.layers.Dropout(0.2),  # Dropout for MC Dropout
        tf.keras.layers.LSTM(50),
        tf.keras.layers.Dropout(0.2),  # Dropout for MC Dropout
        tf.keras.layers.Dense(7)  # 7 outputs, one for each year
    ])
    
    # Compile the model
    baseline_model.compile(optimizer='adam', loss='mean_squared_error')
    
    # Train the model
    history = baseline_model.fit(X_train, y_train, epochs=100, batch_size=16)
    return baseline_model

def get_weighted_metrics(player_data, player_id, season, metrics=['WAR/82', 'MP'], weights=[6, 3, 1]):
    player_data = player_data[player_data['Year'].isin([season-1, season-2, season-3])]

    if len(player_data) == 0:
        return [0] * len(metrics)  # Return None if no data for metrics

    weighted_metrics = []

    for metric in metrics:
        season_metric = []
        season_weights = []

        if season-1 in player_data['Year'].values:
            value = player_data[player_data['Year'] == season-1][metric].sum() if metric != 'MP' else player_data[player_data['Year'] == season-1]['MP'].sum() / player_data[player_data['Year'] == season-1]['G'].sum()
            season_metric.append(value * weights[0])
            season_weights.append(weights[0])

        if season-2 in player_data['Year'].values:
            value = player_data[player_data['Year'] == season-2][metric].sum() if metric != 'MP' else player_data[player_data['Year'] == season-2]['MP'].sum() / player_data[player_data['Year'] == season-2]['G'].sum()
            season_metric.append(value * weights[1])
            season_weights.append(weights[1])

        if season-3 in player_data['Year'].values:
            value = player_data[player_data['Year'] == season-3][metric].sum() if metric != 'MP' else player_data[player_data['Year'] == season-3]['MP'].sum() / player_data[player_data['Year'] == season-3]['G'].sum()
            season_metric.append(value * weights[2])
            season_weights.append(weights[2])

        total_weights = sum(season_weights)
        weighted_avg = sum(season_metric) / total_weights if total_weights > 0 else None
        weighted_metrics.append(weighted_avg)

    return weighted_metrics

def build_team_X_Y(raptor_data, standings_data, season, team_abbreviation):
    team_players = raptor_data[(raptor_data['Franch'] == team_abbreviation) & (raptor_data['Year'] == season) & (raptor_data['Type'] == 'RS')]
    team_players = team_players.sort_values(by='G', ascending=False).head(13)

    X_team = []
    for player_id in team_players['player_ID'].unique():
        player_data = raptor_data[raptor_data['player_ID'] == player_id]
        weighted_metrics = get_weighted_metrics(player_data, player_id, season)
        player_age = player_data[player_data['Year'] == season]['Age'].values[0] if season in player_data['Year'].values else 0
        player_mpg = weighted_metrics[1]
        team_control = player_data[player_data['Year'] == season]['Team Control'].values[0] if 'Team Control' in player_data.columns else 1
        X_team.append([weighted_metrics[0], player_age, player_mpg, team_control])

    while len(X_team) < 13:
        X_team.append([0, 0, 0, 1])

    team_win_pct = standings_data[(standings_data['abbreviation'] == team_abbreviation) & (standings_data['Season_Adjusted'].isin([season + i for i in range(7)]))]['WinPCT'].values
    if len(team_win_pct) < 7:
        team_win_pct = np.pad(team_win_pct, (0, 7 - len(team_win_pct)), 'constant', constant_values=0)

    return X_team, team_win_pct

def process_team_data(team_abbr, season, is_training, raptor_data, standings_data, X_train, Y_train, X_test, Y_test):
    X_team, Y_team = build_team_X_Y(raptor_data, standings_data, season, team_abbr)
    
    # Append data to the corresponding train or test set
    if is_training:
        X_train.append(X_team)
        Y_train.append(Y_team)
    else:
        X_test.append(X_team)
        Y_test.append(Y_team)

def main():
    # Load the original data
    data = pd.read_csv('nba_standings_cleaned_updated.csv')
    raptor_data = pd.read_csv('nba_raptor_with_team_control_updated.csv', encoding='latin1')
    
    data_2024 = data[data['Season_Adjusted'] == 2024]
    western_conf_teams_2024 = data_2024[data_2024['Conference'] == 'West']['abbreviation'].unique()
    eastern_conf_teams_2024 = data_2024[data_2024['Conference'] == 'East']['abbreviation'].unique()

    train_data = data[data['abbreviation'].isin(western_conf_teams_2024)].copy()
    test_data = data[data['abbreviation'].isin(eastern_conf_teams_2024)].copy()
    
    train_data = train_data.drop(columns=['Conference'])
    test_data = test_data.drop(columns=['Conference'])

    # Prepare sequences for LSTM
    train_sequences = create_train_sequences(train_data)
    X_train = np.array([seq[0] for seq in train_sequences])
    y_train = np.array([seq[1] for seq in train_sequences])
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))

    # Build and train LSTM model
    baseline_model = build_lstm_model(X_train, y_train)

    print(X_train)

    # Prepare data for player team control model
    standings_data = data.copy()
    X_train, Y_train, X_test, Y_test = [], [], [], []
    
    for season in range(1991, 2017):
        for team_abbr in standings_data['abbreviation'].unique():
            if team_abbr in western_conf_teams_2024:
                process_team_data(team_abbr, season, is_training=True, raptor_data=raptor_data, standings_data=standings_data, X_train=X_train, Y_train=Y_train, X_test=X_test, Y_test=Y_test)
            elif team_abbr in eastern_conf_teams_2024:
                process_team_data(team_abbr, season, is_training=False, raptor_data=raptor_data, standings_data=standings_data, X_train=X_train, Y_train=Y_train, X_test=X_test, Y_test=Y_test)
    
    X_train = np.array(X_train)
    Y_train = np.array(Y_train)
    X_test = np.array(X_test)
    Y_test = np.array(Y_test)
    
    X_train = np.array(X_train).reshape((X_train.shape[0], X_train.shape[1], 4))
    Y_train = np.array(Y_train)

    print(X_train)
    
    # Build and train player team control model
    player_team_control_model = tf.keras.Sequential([
        tf.keras.layers.LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.LSTM(50),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(7)
    ])
    
    player_team_control_model.compile(optimizer='adam', loss='mean_squared_error')
    player_team_control_model.fit(X_train, Y_train, epochs=100, batch_size=16)

if __name__ == '__main__':
    main()

### Document produit par le groupe de Harry Boisselot et Benoit Catry S5A2
### Etudiants à l'IUT NFC 2025-2026 

### Cette classe permet d'utiliser les fonctions nécessaires à l'IA prédictive.
### Une seule a été utilisée -> predict_weeks_from_date()
class Functions_AI:
    def __init__(self, np, pd, joblib, df, serie, df_feat, X):
        self.np = np
        self.pd = pd
        self.joblib = joblib
        self.df = df
        self.serie = serie
        self.df_features = df_feat
        self.X = X
        

    def predict_weeks_from_date(self, start_date, horizon=4, flu=0):
        """
        start_date : str ou Timestamp -> date à partir de laquelle on veut des prédictions
        horizon    : int -> nombre de semaines à prédire à partir de cette date
        flu        : 0 ou 1 -> indique si on veut simuler un épisode de grippe pour les semaines futures

        La fonction gère automatiquement le cas où start_date > dernière date observée :
        - elle prédit d'abord les semaines manquantes
        - puis renvoie uniquement les 'horizon' semaines à partir de start_date
        """
        start_date = start_date.strip('"').strip("'")

        start_date = self.pd.to_datetime(start_date, format="%Y-%m-%d", errors="raise")

        last_date = self.df_features.index[-1]

        history_y = list(self.df_features["y"].iloc[-12:])

        if start_date <= last_date:
            n_skip = 0
        else:
            delta_days = (start_date - last_date).days
            n_skip = int(self.np.ceil(delta_days / 7))

        total_steps = n_skip + horizon  
        all_dates = []
        all_preds = []
        xgb = self.joblib.load("xgb_hospital.pkl")

        for step in range(1, total_steps + 1):
            current_date = last_date + self.pd.Timedelta(weeks=step)
            all_dates.append(current_date)

            iso = current_date.isocalendar()
            week = int(iso.week)
            month = current_date.month
            year = current_date.year

            flu_flag = int(flu)

            lag_1 = history_y[-1]
            lag_2 = history_y[-2] if len(history_y) >= 2 else lag_1
            lag_3 = history_y[-3] if len(history_y) >= 3 else lag_2
            lag_4 = history_y[-4] if len(history_y) >= 4 else lag_3
            lag_8 = history_y[-8] if len(history_y) >= 8 else history_y[0]
            lag_12 = history_y[-12] if len(history_y) >= 12 else history_y[0]

            last4 = history_y[-4:] if len(history_y) >= 4 else history_y
            roll_mean_4 = self.np.mean(last4)
            roll_std_4  = self.np.std(last4, ddof=0)

            row = self.pd.DataFrame([{
                "week": week,
                "month": month,
                "year": year,
                "flu": flu_flag,
                "lag_1": lag_1,
                "lag_2": lag_2,
                "lag_3": lag_3,
                "lag_4": lag_4,
                "lag_8": lag_8,
                "lag_12": lag_12,
                "roll_mean_4": roll_mean_4,
                "roll_std_4": roll_std_4,
            }], index=[current_date])

            row = row[self.X.columns]

            
            y_hat = xgb.predict(row)[0]
            all_preds.append(y_hat)

            history_y.append(y_hat)

        df_all_future = self.pd.DataFrame({
            "Date": all_dates,
            "Nombre de patients": all_preds
        }).set_index("Date")

        df_from_start = df_all_future[df_all_future.index >= start_date].iloc[:horizon]

        return df_from_start
    
    def predict_from_current(self, start_date, current_y, horizon=4, flu=0):
        """
        start_date : str ou Timestamp -> date correspondant à la semaine actuelle
        current_y  : float/int -> nombre de patients admis cette semaine-là (valeur actuelle)
        horizon    : int -> nombre de semaines à prédire après start_date
        flu        : 0 ou 1 -> simule un épisode de grippe dans les semaines futures

        Idée :
        - On récupère les 11 dernières valeurs observées dans la série
        - On ajoute la valeur actuelle (current_y) comme 12e valeur
        - On prédit ensuite semaine par semaine à partir de start_date + 1 semaine
        """

        start_date = self.pd.to_datetime(start_date)

        history_y = list(self.df_features["y"].iloc[-11:])


        history_y.append(float(current_y))  

        future_dates = []
        future_preds = []

        last_known_date = start_date

        for step in range(1, horizon + 1):
            current_date = last_known_date + self.pd.Timedelta(weeks=step)
            future_dates.append(current_date)

            iso = current_date.isocalendar()
            week = int(iso.week)
            month = current_date.month
            year = current_date.year

            flu_flag = int(flu)

            lag_1 = history_y[-1]
            lag_2 = history_y[-2] if len(history_y) >= 2 else lag_1
            lag_3 = history_y[-3] if len(history_y) >= 3 else lag_2
            lag_4 = history_y[-4] if len(history_y) >= 4 else lag_3
            lag_8 = history_y[-8] if len(history_y) >= 8 else history_y[0]
            lag_12 = history_y[-12] if len(history_y) >= 12 else history_y[0]

            last4 = history_y[-4:] if len(history_y) >= 4 else history_y
            roll_mean_4 = self.np.mean(last4)
            roll_std_4  = self.np.std(last4, ddof=0)

            row = self.pd.DataFrame([{
                "week": week,
                "month": month,
                "year": year,
                "flu": flu_flag,
                "lag_1": lag_1,
                "lag_2": lag_2,
                "lag_3": lag_3,
                "lag_4": lag_4,
                "lag_8": lag_8,
                "lag_12": lag_12,
                "roll_mean_4": roll_mean_4,
                "roll_std_4": roll_std_4,
            }], index=[current_date])

            row = row[self.X.columns]

            xgb = self.joblib.load("xgb_hospital.pkl")
            y_hat = xgb.predict(row)[0]
            future_preds.append(y_hat)

            history_y.append(y_hat)
            
            
        df_future = self.pd.DataFrame({
            "date": future_dates,
            "prediction_patients_admitted": future_preds
        }).set_index("date")

        return df_future

    def predict_next_week_from_today(self, current_y, flu=0):
        """
        current_y : nb de patients de la semaine actuelle
        flu       : 0 ou 1, pour simuler un épisode de grippe sur la semaine future

        Retourne : (next_date, y_hat)
        """

        start_date = self.pd.Timestamp.today().normalize()

        next_date = start_date + self.self.pd.Timedelta(weeks=1)

        history_y = list(self.df_features["y"].iloc[-11:])
        history_y.append(float(current_y))   


        iso = next_date.isocalendar()
        week = int(iso.week)
        month = next_date.month
        year = next_date.year

        flu_flag = int(flu)

        lag_1 = history_y[-1]
        lag_2 = history_y[-2] if len(history_y) >= 2 else lag_1
        lag_3 = history_y[-3] if len(history_y) >= 3 else lag_2
        lag_4 = history_y[-4] if len(history_y) >= 4 else lag_3
        lag_8 = history_y[-8] if len(history_y) >= 8 else history_y[0]
        lag_12 = history_y[-12] if len(history_y) >= 12 else history_y[0]

        last4 = history_y[-4:] if len(history_y) >= 4 else history_y
        roll_mean_4 = self.np.mean(last4)
        roll_std_4  = self.np.std(last4, ddof=0)

        row = self.pd.DataFrame([{
            "week": week,
            "month": month,
            "year": year,
            "flu": flu_flag,
            "lag_1": lag_1,
            "lag_2": lag_2,
            "lag_3": lag_3,
            "lag_4": lag_4,
            "lag_8": lag_8,
            "lag_12": lag_12,
            "roll_mean_4": roll_mean_4,
            "roll_std_4": roll_std_4,
        }], index=[next_date])

        row = row[self.X.columns]
        xgb = self.joblib.load("xgb_hospital.pkl")

        y_hat = xgb.predict(row)[0]

        return next_date, y_hat
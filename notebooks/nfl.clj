^{:nextjournal.clerk/visibility {:code :hide}}
(ns nfl 
  (:require [tablecloth.api :as tc]
            [fastmath.ml.regression :as reg]
            [nextjournal.clerk :as clerk]
            [clojure.instant :as instant]
            [scicloj.kindly.v4.kind :as kind]))

;; Data from https://www.kaggle.com/datasets/nicholasliusontag/nfl-contract-and-draft-data/data

;; Map teams that moved or rebranded themselves
(def team-moves
  {"SDG" "LAC" 
   "STL" "LAR"
   "OAK" "LVR"
   "PHO" "ARI"
   "RAI" "LVR"
   "RAM" "LAR"})

(def salary-data
  (-> "combined_data_2000-2023.csv"
      (tc/dataset {:key-fn keyword})
      (tc/map-columns :tm [:tm] #(or (team-moves %) %))))

(def by-position (->
                   salary-data
                   (tc/group-by :pos)
                   (tc/groups->map)))

;; How does pay correlate to draft position?

(clerk/plotly
  {:data (for [position (keys by-position)
               :let [data (by-position position)]]
           {:x (seq (:pick data))
            :y (seq (:value_norm data))
            :text (seq (:search_key data))
            :type "scatter"
            :mode "markers"
            :name position
            :marker {:size 3}})})

;; How are different positions paid?

(clerk/plotly
  {:data (for [position (keys by-position)]
           {:y (seq (:value_norm (by-position position)))
            :name position
            :type "box" })
   :layout {:showlegend false}})

;; When are different positions drafted?
(clerk/plotly
  {:data (for [position (keys by-position)]
           {:y (seq (:pick (by-position position)))
            :name position
            :type "box" })
   :layout {:showlegend false}})

(def team-mapping
  {"Tennessee Titans" "TEN", "Pittsburgh Steelers" "PIT", "Tampa Bay Buccaneers" "TAM", "Cincinnati Bengals" "CIN", "Washington Football Team" "WAS", "Washington Redskins" "WAS",
   "New York Giants" "NYG", "Buffalo Bills" "BUF", "Baltimore Ravens" "BAL", "Minnesota Vikings" "MIN", "Indianapolis Colts" "IND", "San Diego Chargers" "SDG",
   "St. Louis Rams" "STL" "Arizona Cardinals" "ARI" "Chicago Bears" "CHI" "Houston Texans" "HOU" "Jacksonville Jaguars" "JAX" "Cleveland Browns" "CLE" "Seattle Seahawks" "SEA"
   "Los Angeles Chargers" "LAC" "Oakland Raiders" "OAK" "Green Bay Packers" "GNB" "Philadelphia Eagles" "PHI" "San Francisco 49ers" "SFO" "Denver Broncos" "DEN" "Atlanta Falcons" "ATL"
   "Detroit Lions" "DET" "Kansas City Chiefs" "KAN" "Miami Dolphins" "MIA" "New Orleans Saints" "NOR" "Los Angeles Rams" "LAR" "Carolina Panthers" "CAR" "Dallas Cowboys" "DAL"
   "Las Vegas Raiders" "LVR" "New England Patriots" "NWE" "New York Jets" "NYJ" "Washington Commanders" "WAS"})

;; NFL team data from https://www.kaggle.com/datasets/nickcantalupa/nfl-team-data-2003-2023
(def team-data
  (-> "team_stats_2003_2023.csv"
      (tc/dataset {:key-fn keyword})
      (tc/map-columns :team [:team] team-mapping)
      (tc/map-columns :team [:team] #(or (team-moves %) %))))

;; More complete draft data
(def raw-draft-data
  (-> "NFLDraftHistory.csv"
      (tc/dataset {:key-fn keyword})
      (tc/map-columns :team [:team] #(or (team-moves %) %))))

(def draft-data 
  (let [max-draft (apply max (:pick raw-draft-data))]
    (-> raw-draft-data 
      (tc/group-by [:season :team :category])
      (tc/map-columns :rev-pick [:pick] #(- (inc max-draft) %))
      (tc/aggregate #(reduce + (% :rev-pick)))
      (tc/pivot->wider :category "summary" {:drop-missing? false})
      (tc/replace-missing :all :value 0))))

;; There aren't enough special teams draftees to work with
(def positions ["QB" "RB" "DL" "LB" "DB" "OL" "TE" "WR"])

;; Create a dataset to get the win loss data from a given number of years back
(defn make-data-set
  [years-back]
  (let [wins (-> team-data
                 (tc/map-columns :draft-year [:year] #(- % years-back))
                 (tc/select-columns [:draft-year :team :win_loss_perc]))]
    (-> 
      (tc/inner-join draft-data wins {:left [:team :season] :right [:team :draft-year]})
      (tc/select-columns (conj positions :win_loss_perc)))))

;; Build a linear model
(defn model
  [ds options]
  (reg/lm (:win_loss_perc ds)
          (tc/rows (tc/drop-columns ds [:win_loss_perc]))
          (merge {:names positions} options)))

;; Try models going back from 1 to 10 years
(def evaluations
  (for [years (range 1 10)
        :let [ds (make-data-set years)
              model (model ds {})]]
    {:years-back years
     :rs (:r-squared model)
     :fs (:f-statistic model)}))

;; What do the r squared values look like?
(clerk/plotly {:data [{:y (map :rs evaluations)
                       :x (map :years-back evaluations)}]})

;; Pretty bad but 5 years seems to be best

(def m (model (make-data-set 5) {}))

(clerk/code 
  (-> m
    println
    with-out-str))

;; The only position that is signficant at p <= 0.05 is OL

(clerk/plotly 
  {:data
   [(let [coefficients (rest (:coefficients m))]
      {:y (map :estimate coefficients)
       :error_y {:type 'data'
                 :array (for [c coefficients
                              :let [estimate (:estimate c) interval (:confidence-interval c)]]
                          (- (first interval) estimate))
                 :visible true}
       :x (rest (:names m))
       :type "scatter"
       :mode "markers"})]})

(ns nfl 
  (:require [tablecloth.api :as tc]
            [fastmath.ml.regression :as reg]
            [fastmath.vector :as fmvec]
            [fastmath.stats :as stats]
            [nextjournal.clerk :as clerk]
            [clojure.instant :as instant]
            [scicloj.kindly.v4.kind :as kind]))

;; Data from https://www.kaggle.com/datasets/nicholasliusontag/nfl-contract-and-draft-data/data

;; Map teams that moved or rebranded themselves
{:nextjournal.clerk/visibility {:code :show :result :hide}}
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

{:nextjournal.clerk/visibility {:code :show :result :show}}
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

{:nextjournal.clerk/visibility {:code :show :result :hide}}
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

;; There aren't enough kick returners or long snappers to work with
(def positions ["QB" "RB" "DL" "LB" "DB" "OL" "TE" "WR" "K" "P"])

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
  [ds]
  (reg/lm (:win_loss_perc ds)
          (tc/rows (tc/drop-columns ds [:win_loss_perc]))
          {:names positions}))

;; Try models going back from 1 to 10 years
(def evaluations
  (for [years (range 1 10)
        :let [ds (make-data-set years)
              model (model ds)]]
    {:years-back years
     :rs (:r-squared model)
     :fs (:f-statistic model)}))

;; What do the r squared values look like?
{:nextjournal.clerk/visibility {:code :show :result :show}}
(clerk/plotly {:data [{:y (map :rs evaluations)
                       :x (map :years-back evaluations)}]})

;; Pretty bad but 5 years seems to be best

;; What does that data look like?

^{:nextjournal.clerk/visibility {:code :show :result :hide}}
(def ds (make-data-set 5))

(clerk/code (tc/info ds))

;; Lets build the model

^{:nextjournal.clerk/visibility {:code :show :result :hide}}
(def m (model ds))

(clerk/code 
  (-> m
    println
    with-out-str))

;; The model's F statistic is not statistically signficant, so this is a pretty bad model
;; The only position that is signficant at p <= 0.05 is OL. Kickers have a higher coefficient but there aren't enough of them to be statistically significant.

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

;; What if we look at multiple years
(defn window
  [years-back]
  (->
    (apply tc/concat (for [i (range 1 years-back)]
                       (tc/map-columns draft-data :year [:season] #(+ % i))))
    (tc/group-by [:year :team])
    (tc/aggregate-columns positions #(reduce + %))))

(defn multi-year-dataset
  [years-back]
  (let [drafts (window years-back)]
    (-> 
      (tc/inner-join drafts team-data [:team :year])
      (tc/select-columns (conj positions :win_loss_perc)))))

;; Try models going back from 1 to 10 years
^{:nextjournal.clerk/visibility {:code :show :result :hide}}
(def multi-year-evaluations
  (for [years (range 2 15)
        :let [ds (multi-year-dataset years)
              model (model ds)]]
    {:years-back years
     :rs (:r-squared model)
     :fs (:f-statistic model)}))

;; What do the r squared values look like?
(clerk/plotly {:data [{:y (map :rs multi-year-evaluations)
                       :x (map :years-back multi-year-evaluations)}]})

;; Looking at the draft positions for the past 12 years explains about 5% of the win percentage. Lets go with that

^{:nextjournal.clerk/visibility {:code :show :result :hide}}
(def multi-year-ds (multi-year-dataset 12))

(clerk/code (tc/info multi-year-ds))

^{:nextjournal.clerk/visibility {:code :show :result :hide}}
(def multi-year-model (model multi-year-ds))

(clerk/code 
  (-> multi-year-model
    println
    with-out-str))

;; Now line backers and kickers are also positive and statistically significant at p <= 0.05. Kickers have the highest coefficient but their standard error remains fairly large. The Offensive Linemen have the second highest coefficient and are significant at p <= 0.0001.
;; Also the model now has a significant F statistic.

(clerk/plotly 
  {:data
   [(let [coefficients (rest (:coefficients multi-year-model))]
      {:y (map :estimate coefficients)
       :error_y {:type 'data'
                 :array (for [c coefficients
                              :let [estimate (:estimate c) interval (:confidence-interval c)]]
                          (- (first interval) estimate))
                 :visible true}
       :x (rest (:names m))
       :type "scatter"
       :mode "markers"})]})

;; Lets also look at the correlations

(clerk/plotly
  {:data [{:x positions
           :y (for [p positions] (stats/correlation (:win_loss_perc multi-year-ds) (multi-year-ds p)))
           :type "bar"
           }]
   })

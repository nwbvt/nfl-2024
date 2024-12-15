(require '[nextjournal.clerk :as clerk])

(clerk/serve! {:watch-paths ["notebooks" "src"] :host "0.0.0.0"})

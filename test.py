from lda import TopicModeling

lda_modeling = TopicModeling()
lda_modeling.setTopicNum(2)

paragraph = "The Demand for Foreign Trade. Beginning in the early 19th century, Westerners tried to convince the Japanese to open their ports to trade. British, French, Russian, and American officials occasionally anchored off the Japanese coast. Like China, however, Japan repeatedly refused to receive them. Then, in 1853, U.S. Commodore Matthew Perry took four ships into what is now Tokyo Harbor. These massive black wooden ships powered by steam astounded the Japanese. The ships’ cannons also shocked them. The Tokugawa shogun realized he had no choice but to receive Perry and the letter Perry had brought from U.S. president Millard Fillmore."

print(lda_modeling.groupSentence(paragraph))
distanz_alle <- c(1, 3, 5, 2, 3, 3, 3, 2, 0, 1, 3, 4, 2, 1, 5, 
                                  1, 1, 1, 1, 5, 2, 
            2, 1, 0, 4, 3, 1, 6, 1, 3, 1, 3, 5, 5, 4, 3, 2, 1, 0, 1, 3, 1, 0, 4, 
            1, 0, 3, 3, 1, 1, 2, 6, 1, 2, 2, 1, 1, 2, 0, 2, 3, 3, 2, 2, 3, 1, 1, 
            3, 2, 2, 2, 4, 2, 3, 1, 0, 4, 2, 2, 3, 7, 1, 2, 2, 4, 3, 1, 2, 0, 1, 
            0, 1, 2, 1, 1, 4, 1, 2, 3, 1, 1, 2, 6, 1, 2, 4, 2, 7, 3, 4, 6, 4, 5, 
            3, 4, 4, 4, 1, 3, 5, 4, 1, 1, 1, 5, 0, 1, 0, 3, 1, 1, 0, 2, 1, 3, 2, 
            4, 4, 2, 2, 1, 2, 3, 1, 2, 3, 2, 4, 7, 0, 3, 7, 4, 4, 1, 2, 3, 4, 3,
            5, 3, 2, 4, 7, 4, 5, 6, 1, 3, 4, 2, 4, 7, 4, 3, 1, 3, 1, 0, 3, 3, 4, 
              3, 1, 1, 5, 5, 5, 0, 8, 1, 4, 0, 0, 4, 3, 6, 2, 3, 6, 3, 4, 0, 5, 
            4, 4, 6, 2, 2, 2, 2, 0, 1, 3, 3, 0, 5, 2, 4, 1, 0, 2, 1, 4, 4, 2, 1, 
            0, 6, 1, 1, 2, 0, 2, 4, 2, 0, 1, 3, 1, 5, 1, 2, 1, 1, 1, 5, 5, 6, 1, 
            6, 3, 3, 5, 0, 3, 3, 6, 3, 2, 4, 3, 2, 2, 5, 6, 1, 4, 5, 5, 1, 2, 5, 
            5, 1, 2, 6, 2, 2, 3, 4, 6, 2, 1, 1, 4, 5, 4, 4, 2, 3, 1, 3, 4, 7, 1, 
            3, 2, 1, 6, 2, 7, 5, 4, 2, 6, 4, 4, 4, 5, 7, 5, 1, 2, 7, 3, 6, 7, 3, 
            4, 3, 5, 2, 4, 2, 2, 6, 6, 3, 3, 2, 1, 1, 1, 2, 2, 1, 0, 1, 1, 1, 2, 
            1, 2, 1, 5, 0, 1, 2, 0, 2, 0, 6, 5, 1, 2, 3, 2, 2, 0, 0, 2, 1, 1, 1, 
            4, 3, 4, 6, 2, 0, 3, 3, 4, 1, 1, 1, 3, 3, 2, 3, 0, 5, 2, 4, 5, 1, 2, 
            3, 3, 4, 0, 1, 1, 0, 1, 1, 4, 0, 4, 3, 3, 4, 1, 1, 1, 2, 2, 2, 2, 4, 
            1, 2, 6, 1, 2, 4, 7, 4, 1, 2, 2, 2, 3, 3, 2, 1, 1, 2, 1, 1, 1, 1, 2, 
            2, 2, 1, 1, 2, 2, 2, 4, 1, 4, 2, 2, 5, 5, 4, 6, 0, 0, 0, 1, 3, 0, 3, 
            3, 0, 3, 2, 3, 2, 3, 3, 5, 0, 1, 1, 4, 1, 2, 1, 0, 1, 1, 0, 0, 1, 2,  
            0, 3, 1, 0, 4, 0, 2, 1, 4, 3, 5, 0, 4, 0, 0, 0, 1, 2, 4, 3, 3, 2, 0, 
            2, 0, 2, 0, 1, 2, 0, 2, 1, 1, 0, 0, 0, 1, 1, 1, 4, 2, 1, 4, 4, 6, 1, 
            3, 3, 6, 4, 3, 1, 3, 1, 0, 1, 3, 4, 3, 4, 2, 3, 6, 3, 2, 6, 2, 1, 3, 
            2, 4, 4, 4, 4, 2, 4, 1, 2, 2, 3, 1, 3, 1, 2, 0, 1, 1, 0, 8, 2, 3, 4, 
            2, 3, 1, 0, 2, 2, 2, 1, 6, 1, 1, 2, 0, 2, 2, 5, 5, 1, 3, 3, 2, 3, 4, 
            4, 6, 0, 1, 3, 4, 2, 4, 4, 3, 1, 2, 1, 5, 3, 2, 5, 4, 3, 1, 3, 4, 4, 
            4, 3, 5, 2, 4, 2, 4, 2)

distanz_uni <- c(1, 3, 3, 3, 2, 0, 3, 4, 2, 1, 5, 1, 1, 5, 2, 1, 1, 6, 1, 3, 1, 
                5, 5, 4, 3, 2, 1, 0, 3, 1, 0, 4, 1, 0, 3, 3, 1, 1, 2, 6, 1, 2, 2, 
                1, 1, 2, 0, 2, 3, 3, 2, 2, 3, 1, 1, 3, 2, 2, 2, 4, 2, 3, 1, 0, 4, 
                2, 2, 3, 1, 2, 2, 4, 3, 1, 0, 1, 0, 1, 2, 1, 1, 4, 2, 1, 2, 6, 1, 
                2, 4, 2, 4, 6, 4, 5, 4, 4, 1, 3, 5, 4, 1, 1, 1, 5, 0, 1, 0, 3, 1,
                1, 0, 2, 2, 4, 4, 2, 1, 2, 3, 2, 4, 7, 0, 3, 7, 4, 4, 1, 2, 3, 4, 
                3, 5, 3, 4, 7, 4, 5, 6, 1, 3, 4, 4, 3, 1, 1, 0, 3, 3, 4, 3, 1, 1, 
                5, 1, 2, 2, 2, 0, 1, 3, 3, 0, 2, 4, 1, 0, 1, 4, 4, 1, 6, 2, 0, 2, 
                2, 0, 3, 1, 5, 1, 2, 1, 4, 4, 2, 3, 1, 3, 4, 7, 1, 3, 2, 1, 6, 2, 
                7, 5, 4, 2, 6, 4, 4, 4, 5, 7, 5, 1, 2, 7, 3, 6, 7, 3, 4, 3, 5, 2, 
                4, 2, 2, 6, 6, 3, 3, 2, 1, 1, 1, 2, 2, 1, 0, 1, 1, 1, 2, 1, 2, 1, 
                5, 0, 1, 2, 2, 6, 3, 2, 2, 0, 0, 2, 1, 1, 4, 3, 3, 4, 1, 1, 3, 2, 
                1, 2, 3, 3, 4, 1, 1, 0, 1, 1, 4, 0, 4, 3, 3, 4, 1, 1, 1, 2, 2, 2, 
                2, 4, 1, 2, 6, 1, 2, 4, 7, 4, 1, 2, 2, 2, 3, 3, 2, 1, 1, 2, 1, 1, 
                1, 1, 2, 2, 2, 1, 1, 2, 2, 2, 4, 1, 4, 2, 
                2, 5, 5, 0, 0, 0, 1, 3, 0, 3, 3, 0, 3, 2, 3, 2, 3, 3, 5, 0, 1, 1, 
                4, 1, 2, 1, 0, 1, 1, 0, 0, 1, 1, 0, 3, 1, 0, 4, 2, 1, 4, 3, 0, 4, 
                0, 0, 0, 1, 2, 4, 3, 3, 2, 0, 2, 0, 2, 0, 1, 2, 0, 2, 1, 1, 0, 0, 
                0, 1, 1, 1, 4, 2, 1, 1, 3, 4, 3, 3, 1, 0, 3, 4, 3, 3, 6, 3, 2, 2, 
                1, 3, 2, 4, 4, 4, 2, 4, 1, 2, 3, 2, 1, 0, 2, 3, 4, 2, 3, 0, 2, 2, 
                2, 1, 6, 1, 1, 2)
distanz_eu_ro <- c(1, 4, 1, 1, 1, 5, 0, 1, 0, 3, 1, 1, 0, 2, 2, 2, 1, 4, 2, 1, 1, 
                   6, 0, 4, 4, 2, 3, 1, 3, 4, 7, 1, 3, 2, 1, 6, 2, 7, 5, 4, 2, 6, 
                   4, 4, 4, 5, 7, 5, 1, 2, 7, 3, 6, 7, 3, 4, 3, 5, 2, 4, 2, 2, 6, 
                   6, 3, 3, 2, 2, 0, 0, 2, 2, 1, 0, 4, 0, 0, 0, 1, 2, 4, 3, 3, 2, 
                   0, 2, 0, 2, 0, 1, 2, 0, 2, 1, 1, 0, 0, 0, 1, 1, 1, 4, 2, 1, 4, 
                   2, 3, 2, 1, 3, 2)

table(distanz_alle)
table(distanz_uni)
table(distanz_eu_ro)

tab_1 <- prop.table(table(distanz_alle))
tab_2 <- prop.table(table(distanz_uni))
tab_3 <- prop.table(table(distanz_eu_ro))

sum(tab_1[1:3])
sum(tab_2[1:3])
sum(tab_3[1:3])

hist(distanz_alle, freq=FALSE, main = "Uni-Datensatz + eigene Bilder", 
     ylab = "relative Häufigkeit", xlab = "Levenshtein-Distanz", breaks = 0:9, 
     right = FALSE, ylim = c(0, 0.3), cex.axis = 1.5, cex.lab=1.5, cex.main=1.5)
hist(distanz_uni, freq=FALSE, main = "Uni-Datensatz", 
     ylab = "relative Häufigkeit", xlab = "Levenshtein-Distanz", breaks = 0:9, 
     right = FALSE, ylim = c(0, 0.3), cex.axis = 1.5, cex.lab=1.5, cex.main=1.5)
hist(distanz_eu_ro, freq=FALSE, main = "EU+RO", 
     ylab = "relative Häufigkeit", xlab = "Levenshtein-Distanz", breaks = 0:9, 
     right = FALSE, ylim = c(0, 0.3), cex.axis = 1.5, cex.lab=1.5, cex.main=1.5)

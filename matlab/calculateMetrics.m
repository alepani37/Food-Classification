function [macroF1, precision, recall, accuracy] = calculateMetrics(confusionMatrix, classes)
    % Calcola precision, recall e F1-score per ciascuna classe
    numClasses = numel(classes);
    precision = zeros(numClasses, 1);
    recall = zeros(numClasses, 1);
    f1Scores = zeros(numClasses, 1);
    
    totalTP = 0; % Inizializza la variabile per il calcolo dell'accuratezza
    totalSamples = sum(confusionMatrix(:)); % Numero totale di campioni
    
    for i = 1:numClasses
        TP = confusionMatrix(i, i); % True Positive
        FP = sum(confusionMatrix(:, i)) - TP; % False Positive
        FN = sum(confusionMatrix(i, :)) - TP; % False Negative
        
        % Aggiorna il numero totale di True Positives
        totalTP = totalTP + TP;
        
        if TP + FP == 0
            precision(i) = 0;
        else
            precision(i) = TP / (TP + FP);
        end
        
        if TP + FN == 0
            recall(i) = 0;
        else
            recall(i) = TP / (TP + FN);
        end
        
        if precision(i) + recall(i) == 0
            f1Scores(i) = 0;
        else
            f1Scores(i) = 2 * (precision(i) * recall(i)) / (precision(i) + recall(i));
        end
    end
    
    % Calcola F1-score macro
    macroF1 = mean(f1Scores);
    
    % Calcola l'accuratezza
    accuracy = totalTP / totalSamples;
    recall  = mean(recall);
    precision = mean(precision);
end

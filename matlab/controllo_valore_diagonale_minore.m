% Legge la matrice di confusione dal file CSV
filename = 'M.csv'; % Sostituisci con il nome del tuo file CSV
confusionMatrix = csvread(filename);

% Controlla i valori sulla diagonale principale
diagonalValues = diag(confusionMatrix);

% Trova il valore minimo nella diagonale
minValue = min(diagonalValues);
%diagonalValues

% Trova le colonne (classi) associate al valore minimo
minColumns = find(diagonalValues == minValue);

% Stampa il risultato
fprintf('Il valore minimo sulla diagonale è: %d\n', minValue);
fprintf('Associato alla colonna/alle colonne: %s\n', num2str(minColumns'));


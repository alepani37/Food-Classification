%Script per aggiunger righe a un csv in merito ad una prima analisi del
%sottinsieme di classi migliore per il dataset di immagini di cibo
% cellarray_di_valori = {'valore1', 'valore2', 'valore3'};


function debug_inserisci_righe_al_csv(filename, cellarray_di_valori)
% Nome del file CSV
%filename = 'primaanalisi.csv';

% Dati da aggiungere (come cell array)
%newRow = {'valore1', 'valore2', 'valore3'}; % Aggiungi qui i valori per ogni campo

% Controlla se il file esiste già
if isfile(filename)
    data = readcell(filename); % Se il file esiste, leggi il contenuto attuale
else
    % Se il file non esiste, crea una cella vuota per i dati
    data = {};
end

% Aggiungi la nuova riga ai dati esistenti
data = [data; cellarray_di_valori];

% Scrivi i dati aggiornati nel file CSV
writecell(data, filename);

disp(['Riga aggiunta al file ' filename]);

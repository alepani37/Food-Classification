% debug_count_files_in_subdirectories("../img/prova_resized")
function debug_count_files_in_subdirectories(parentDir)
    % Verifica se la directory specificata esiste
    if ~isfolder(parentDir)
        error('La directory specificata non esiste.');
    end

    % Ottiene la lista di tutte le sottodirectory
    dirInfo = dir(parentDir);
    isSubdir = [dirInfo.isdir];
    subDirs = dirInfo(isSubdir);
    
    % Rimuove le directory '.' e '..'
    subDirs = subDirs(~ismember({subDirs.name}, {'.', '..'}));
    
    % Inizializza una cella per contenere i risultati
    results = cell(length(subDirs), 2);

    % Conta i file in ciascuna sottodirectory
    for k = 1:length(subDirs)
        subDirPath = fullfile(parentDir, subDirs(k).name);
        fileCount = numel(dir(fullfile(subDirPath, '*'))) - 2; % sottrae 2 per rimuovere '.' e '..'

        %siccome ci sono anche i file .sift ormai tolgo anche quelli
        fileCount = fileCount / 2;
        
        % Salva i risultati
        results{k, 1} = subDirs(k).name;
        results{k, 2} = fileCount;
    end

    
    % Stampa i risultati
    fprintf('Numero di file nelle sottodirectory di %s:\n', parentDir);
    for k = 1:size(results, 1)
        fprintf('%s: %d file\n', results{k, 1}, results{k, 2});
    end

    
end

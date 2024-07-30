%test_sift_varie_combinazioni

%Script per lanciare diversi test assieme mediante varianti della SIFT
% Directory principale contenente le sottocartelle con le immagini
mainDir = '../img/prova_resized_bn/';

% Ottieni una lista delle sottocartelle
subFolders = dir(mainDir);

% Filtra le sottocartelle per rimuovere le cartelle "." e ".."
subFolders = subFolders([subFolders.isdir]);
subFolders = subFolders(~ismember({subFolders.name}, {'.', '..'}));

% Scorri ogni sottocartella
for k = 1:length(subFolders)
    subDirPath = fullfile(mainDir, subFolders(k).name);
    
    % Ottieni una lista di tutte le immagini nella sottocartella
    imageFiles = dir(fullfile(subDirPath, '*.*'));
   
    imageFiles = imageFiles(~[imageFiles.isdir]);

    % Scorri ogni immagine
    parfor i = 1:length(imageFiles)
        % Percorso completo dell'immagine
        imagePath = fullfile(subDirPath, imageFiles(i).name);
        
        % Leggi l'immagine
        img = imread(imagePath);
        
        % Controlla se l'immagine è già in scala di grigi
        if size(img, 3) == 3
            % Converti l'immagine in scala di grigi
            grayImg = rgb2gray(img);
            
            % Salva l'immagine in scala di grigi nello stesso percorso
            imwrite(grayImg, imagePath);
        end
    end
end

disp('Tutte le immagini sono state convertite in scala di grigi.');

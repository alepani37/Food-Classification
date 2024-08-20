
% Funzione per caricare e assegnare i dati
function [x, y, rad] = load_sift_data(filename)
    data = load(filename);
    if isstruct(data)
        % Se è una struttura, estrae i campi
        vars = fieldnames(data);
        if length(vars) >= 3
            x = data.(vars{1});
            y = data.(vars{2});
            rad = data.(vars{3});
        else
            error('File does not contain expected number of variables');
        end
    elseif isnumeric(data)
        % Se è una matrice numerica, presuppone che le colonne siano c, r e rad
        x = data(:, 1);
        y = data(:, 2);
        rad = data(:, 3);
    else
        error('Unsupported data format in the .sift_pyramid file');
    end
end
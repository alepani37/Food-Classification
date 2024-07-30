%This function takes as input the directory containing the dataset.
%For example if we have 4 categories, say airplanes,faces,motorbikes and
%cars directory structure should be:   ./caltech4
%                                      ./caltech4/airplanes
%                                      ./caltech4/faces
%                                      ./caltech4/motorbikes
%                                      ./caltech4/cars
% This functions creates a random split of the dataset. For each category 
% selects Ntrain training images and min(N-Ntrain,Ntest) test images, where
% N is the amount of images of a given category.
% outputs a structure array with the following fields
%    n_images: 1074
%    classname: 'airplanes'; 
%    files: {1x1074 cell}; cell array with file names withouth path, e.g. img_100.jpg
%    train_id: [1x1074 logical]; Boolean array indicating training files
%    test_id: [1x1074 logical];  Boolean array indicating test files                                   
function data = create_dataset_split_structure_from_unbalanced_sets_val(main_dir,Ntrain,Nval,Ntest,file_ext, numero_di_img_da_considerare)
% CREATE_DATASET_SPLIT_STRUCTURE crea un dataset da sorgenti sbilanciate di
% immagini
    %main_dir = fullfile(basepath, 'img', dataset_dir)
    category_dirs = dir(main_dir);
 
    %remove '..' and '.' directories
    category_dirs(~cellfun(@isempty, regexp({category_dirs.name}, '\.*')))=[];
    category_dirs(strcmp({category_dirs.name},'split.mat'))=[]; 
    
    %Conteggio per avere la classe sbilanciata con meno immagini
    numero_file_minimo_per_classe = realmax;
    for c = 1:length(category_dirs)
        imgdir = dir(fullfile(main_dir,category_dirs(c).name, ['*.' file_ext]));
        if numero_file_minimo_per_classe > length(imgdir)
            numero_file_minimo_per_classe = length(imgdir);
        end
    end 
    
    for c = 1:length(category_dirs)
        fprintf("iterazione %d di %d \n", c, length(category_dirs))
        if isdir(fullfile(main_dir,category_dirs(c).name)) ...
                && ~strcmp(category_dirs(c).name,'.') ...
                && ~strcmp(category_dirs(c).name,'..')
            
            imgdir = dir(fullfile(main_dir,category_dirs(c).name, ['*.' file_ext]));
            %Questo permette di non dover dare un nome specifico ai file vari, basta
            %solo che abbiano tutti la stessa estensione
            
            try
                imgdir = imgdir(1:numero_di_img_da_considerare);% prendiamo solo i primi n 
                ids = randperm(length(imgdir));
                
            catch 
                warning("Numero di immagini della categoria meno rappresentata: %d", numero_file_minimo_per_classe);
                warning("Numero di immagini scelte per ogni classe del dataset: %d", Ntrain);
                error("Hai scelto troppi file per fare il training oppure hai troppe poche immagini di una certa categoria");
            end
            
            data(c).n_images = length(imgdir);
            data(c).classname = category_dirs(c).name;
            data(c).files = {imgdir(:).name};
    
    
            data(c).train_id = false(1,data(c).n_images);
            data(c).train_id(ids(1:Ntrain))=true;
    
            data(c).val_id = false(1,data(c).n_images);
            data(c).val_id(ids(Ntrain+1:Ntrain+Nval))=true;
    
            data(c).test_id = false(1,data(c).n_images);
            data(c).test_id(ids(Ntrain+Nval+1:Ntrain+Nval+min(Ntest,data(c).n_images-Ntrain-Nval)))=true;

        end
    end
end

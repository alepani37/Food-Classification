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
function data = create_dataset_split_structure(main_dir,Ntrain,Ntest,file_ext)
% CREATE_DATASET_SPLIT_STRUCTURE crea un dataset da sorgenti sbilanciate di
% immagini

    category_dirs = dir(main_dir);
 
    %remove '..' and '.' directories
    category_dirs(~cellfun(@isempty, regexp({category_dirs.name}, '\.*')))=[];
    category_dirs(strcmp({category_dirs.name},'split.mat'))=[]; 

    numero_file_minimo_per_classe = realmax;
    for c = 1:length(category_dirs)
        imgdir = dir(fullfile(main_dir,category_dirs(c).name, ['*.' file_ext]));
        if numero_file_minimo_per_classe > length(imgdir)
            numero_file_minimo_per_classe = length(imgdir);
        end
    end 
    
    for c = 1:length(category_dirs)
        
        if isdir(fullfile(main_dir,category_dirs(c).name)) ...
                && ~strcmp(category_dirs(c).name,'.') ...
                && ~strcmp(category_dirs(c).name,'..')
            imgdir = dir(fullfile(main_dir,category_dirs(c).name, ['*.' file_ext]));
            %Questo permette di non dover dare un nome specifico ai file vari, basta
            %solo che abbiano tutti la stessa estensione
            
            ids = randperm(length(imgdir));
            try
                ids = ids(1:Ntrain); %prendiamo solo i primi n
                imgdir = imgdir(1:Ntrain);% prendiamo solo i primi n 
            catch 
                warning("Numero di immagini della categoria meno rappresentata: %d", numero_file_minimo_per_classe);
                warning("Numero di immagini scelte per ogni classe del dataset: %d", Ntrain);
                raise("Hai scelto troppi file per fare il training oppure hai troppe poche immagini di una certa categoria");
            end
            %if number_of_file_to_load >= count %ho aggiunto questo controllo altrimenti crasha
            %length(imgdir)
            data(c).n_images = length(imgdir);
            data(c).classname = category_dirs(c).name;
            data(c).files = {imgdir(:).name};
            
            data(c).train_id = false(1,data(c).n_images);
            %ids(1:Ntrain)
            data(c).train_id(ids(1:Ntrain))=true;
            
            data(c).test_id = false(1,data(c).n_images);
            data(c).test_id(ids(Ntrain+1:Ntrain+min(Ntest,data(c).n_images-Ntrain)))=true;

        end
    end
end

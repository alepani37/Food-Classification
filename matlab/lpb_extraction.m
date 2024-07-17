function [trainLBP,testLBP] = lpb_extraction(data,num_classes,num_istance_per_class_train,num_instance_per_class_test,info)
    
    trainLBP = cell(num_classes,num_istance_per_class_train);
    testLBP = cell(num_classes,num_instance_per_class_test);

    for i = 1 : length(data)
        %estrazione delle LBP per ogni immagine di train

        %estrazione immagini di train della i-esima classe
        img = data(i).files(data(i).train_id);
        for j = 1 : length(img)
            
            im = imread(fullfile(info.base,info.first,info.dsdir,data(i).classname,img(j)));
            %conversione dell'immagine in grayscale
            im = rgb2gray(im); 

            %lbp extraction
            lbp_img = extractLBPFeatures(im);
            trainLBP{i,j} = lbp_img;
        end

        %estrazione immgini di test della i-esima classe
        img = data(i).files(data(i).test_id);
        for j = 1 : length(img)
            
            im = imread(fullfile(info.base,info.first,info.dsdir,data(i).classname,img(j)));
            %conversione dell'immagine in grayscale
            im = rgb2gray(im); 

            %lbp extraction
            lbp_img = extractLBPFeatures(im);
            testLBP{i,j} = lbp_img;
        end
    end
    fprintf("Estrazione delle feature LBP completata correttamente\n");
end
function [trainLBP, valLBP, testLBP] = lbp_val_extraction(data, num_classes, num_istance_per_class_train, num_instance_per_class_val, num_instance_per_class_test, info)
    
    % Inizializza trainLBP, valLBP, e testLBP come array di strutture
    trainLBP = struct('filename', {}, 'class', {}, 'hist', {});
    valLBP = struct('filename', {}, 'class', {}, 'hist', {});
    testLBP = struct('filename', {}, 'class', {}, 'hist', {});

    trainLBP_ = cell(num_classes, num_istance_per_class_train);
    valLBP_ = cell(num_classes, num_instance_per_class_val);
    testLBP_ = cell(num_classes, num_instance_per_class_test);

    for i = 1 : length(data)
        disp(length(data))
        i
        
        % Estrazione immagini di train della i-esima classe
        img = data(i).files(data(i).train_id);
        for j = 1 : length(img)
            im = imread(fullfile(info.base, info.first, info.dsdir, data(i).classname, img(j)));
            % Conversione dell'immagine in grayscale
            im = rgb2gray(im); 

            % LBP extraction
            lbp_img = extractLBPFeatures(im);
            fname = fullfile(info.base, info.first, info.dsdir, data(i).classname, img(j));
            trainLBP_{i, j}.filename = regexprep(fname, ['.' info.desc_name], '.jpg');
            trainLBP_{i, j}.hist = lbp_img;
        end

        % Estrazione immagini di validazione della i-esima classe
        img = data(i).files(data(i).val_id);
        for j = 1 : length(img)
            im = imread(fullfile(info.base, info.first, info.dsdir, data(i).classname, img(j)));
            % Conversione dell'immagine in grayscale
            im = rgb2gray(im); 

            % LBP extraction
            lbp_img = extractLBPFeatures(im);
            fname = fullfile(info.base, info.first, info.dsdir, data(i).classname, img(j));
            valLBP_{i, j}.filename = regexprep(fname, ['.' info.desc_name], '.jpg');
            valLBP_{i, j}.hist = lbp_img;
        end

        % Estrazione immagini di test della i-esima classe
        img = data(i).files(data(i).test_id);
        for j = 1 : length(img)
            im = imread(fullfile(info.base, info.first, info.dsdir, data(i).classname, img(j)));
            % Conversione dell'immagine in grayscale
            im = rgb2gray(im); 

            % LBP extraction
            lbp_img = extractLBPFeatures(im);
            fname = fullfile(info.base, info.first, info.dsdir, data(i).classname, img(j));
            testLBP_{i, j}.filename = regexprep(fname, ['.' info.desc_name], '.jpg');
            testLBP_{i, j}.hist = lbp_img;
        end
    end
    
    num = 1;
    for i = 1 : size(trainLBP_, 1)
        for j = 1 : size(trainLBP_, 2)
            trainLBP(num).filename = trainLBP_{i, j}.filename;
            trainLBP(num).class = i;
            trainLBP(num).hist = trainLBP_{i, j}.hist;
            num = num + 1; 
        end
    end

    num = 1;
    for i = 1 : size(valLBP_, 1)
        for j = 1 : size(valLBP_, 2)
            valLBP(num).filename = valLBP_{i, j}.filename;
            valLBP(num).class = i;
            valLBP(num).hist = valLBP_{i, j}.hist;
            num = num + 1; 
        end
    end

    num = 1;
    for i = 1 : size(testLBP_, 1)
        for j = 1 : size(testLBP_, 2)
            testLBP(num).filename = testLBP_{i, j}.filename;
            testLBP(num).class = i;
            testLBP(num).hist = testLBP_{i, j}.hist;
            num = num + 1; 
        end
    end

    fprintf("Estrazione delle feature LBP completata correttamente\n");
end

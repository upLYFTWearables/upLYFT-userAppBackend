function convertLevelgroundJPIMU(originalDataDir, newDataDir)
%CONVERTLEVELGROUNDJPIMU  Recursively find .mat files under ABxx/<date>/levelground/jp or imu,
%                         load the table variable 'data' if it exists, convert to array + varNames,
%                         then save them (v7) in a parallel folder structure under `newDataDir`.
%
% Usage:
%   convertLevelgroundJPIMU('/Users/.../Data', '/Users/.../Data2')

    if ~isfolder(newDataDir)
        mkdir(newDataDir);
    end
    
    % --- Collect only the jp and imu .mat files under levelground ---
    jpFiles = dir(fullfile(originalDataDir, 'AB*', '*', 'levelground', 'jp', '*.mat'));
    imuFiles = dir(fullfile(originalDataDir, 'AB*', '*', 'levelground', 'imu', '*.mat'));
    gcLeftFiles = dir(fullfile(originalDataDir, 'AB*', '*', 'levelground', 'gcLeft', '*.mat'));
    gcRightFiles = dir(fullfile(originalDataDir, 'AB*', '*', 'levelground', 'gcRight', '*.mat'));
    
    % Combine them into one list
    matFiles = [jpFiles; imuFiles;gcLeftFiles;gcRightFiles];
    
    fprintf('Found %d .mat files in jp/imu levelground subfolders.\n', numel(matFiles));

    for iFile = 1:numel(matFiles)
        oldFolderPath = matFiles(iFile).folder;   % e.g.: .../AB06/10_09_18/levelground/jp
        oldFileName   = matFiles(iFile).name;     % e.g.: levelground_ccw_fast_01_01.mat
        oldFullPath   = fullfile(oldFolderPath, oldFileName);
        
        % --- Build the parallel subfolder path in `newDataDir` ---
        relativeFolder = strrep(oldFolderPath, originalDataDir, '');
        % Make sure we remove any leading file separator if it exists
        if startsWith(relativeFolder, filesep)
            relativeFolder = relativeFolder(2:end);
        end
        
        newFolderPath  = fullfile(newDataDir, relativeFolder);
        if ~isfolder(newFolderPath)
            mkdir(newFolderPath);
        end
        
        newFullPath = fullfile(newFolderPath, oldFileName);
        
        % --- Load the old .mat file from the original data directory ---
        S = load(oldFullPath);
        
        % Check if there's a variable named 'data' and it's a table
        if isfield(S, 'data') && istable(S.data)
            % Convert table -> numeric array
            dataMatrix = table2array(S.data);
            % Grab variable (column) names
            varNames = S.data.Properties.VariableNames;
            
            % Save the array + varNames in v7 format
            save(newFullPath, 'dataMatrix', 'varNames', '-v7');
        else
            % If there's no 'data' table, just copy everything
            % into a new MAT file, ensuring it's v7 or earlier
            % (You can adapt if you want a different behavior.)
            save(newFullPath, '-struct', 'S', '-v7');
        end
    end
    
    fprintf('Done. Processed %d .mat files.\n', numel(matFiles));
end

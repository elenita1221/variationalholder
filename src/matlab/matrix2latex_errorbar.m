function matrix2latex_errorbar(matrix1, matrix2, filename, varargin)
% an extension of matrix2latex that takes in two matrices and outputs
% errorbars as well
% ij^{th} entry of the table is matrix1(i,j) \pm matrix2(i,j)
% see matrix2latex.m for more info

rowLabels = [];
colLabels = [];
alignment = 'c';
format = [];
textsize = [];
%     if (rem(nargin,2) == 1 || nargin < 2)
%         error('matrix2latex: ', 'Incorrect number of arguments to %s.', mfilename);
%     end

okargs = {'rowlabels','columnlabels', 'alignment', 'format', 'size'};
for j=1:2:(nargin-3)
    pname = varargin{j};
    pval = varargin{j+1};
    k = strmatch(lower(pname), okargs);
    if isempty(k)
        error('matrix2latex: ', 'Unknown parameter name: %s.', pname);
    elseif length(k)>1
        error('matrix2latex: ', 'Ambiguous parameter name: %s.', pname);
    else
        switch(k)
            case 1  % rowlabels
                rowLabels = pval;
                if isnumeric(rowLabels)
                    rowLabels = cellstr(num2str(rowLabels(:)));
                end
            case 2  % column labels
                colLabels = pval;
                if isnumeric(colLabels)
                    colLabels = cellstr(num2str(colLabels(:)));
                end
            case 3  % alignment
                alignment = lower(pval);
                if alignment == 'right'
                    alignment = 'r';
                end
                if alignment == 'left'
                    alignment = 'l';
                end
                if alignment == 'center'
                    alignment = 'c';
                end
                if alignment ~= 'l' && alignment ~= 'c' && alignment ~= 'r'
                    alignment = 'l';
                    warning('matrix2latex: ', 'Unkown alignment. (Set it to \''left\''.)');
                end
            case 4  % format
                format = lower(pval);
            case 5  % format
                textsize = pval;
        end
    end
end

fid = fopen(filename, 'w');

width = size(matrix1, 2);
height = size(matrix1, 1);

if isnumeric(matrix1)
    matrix1 = num2cell(matrix1);
    for h=1:height
        for w=1:width
            if(~isempty(format))
                matrix1{h, w} = num2str(matrix1{h, w}, format);
            else
                matrix1{h, w} = num2str(matrix1{h, w});
            end
        end
    end
end

if isnumeric(matrix2)
    matrix2 = num2cell(matrix2);
    for h=1:height
        for w=1:width
            if(~isempty(format))
                matrix2{h, w} = num2str(matrix2{h, w}, format);
            else
                matrix2{h, w} = num2str(matrix2{h, w});
            end
        end
    end
end


if(~isempty(textsize))
    fprintf(fid, '\\begin{%s}', textsize);
end

fprintf(fid, '\\begin{tabular}{|');

if(~isempty(rowLabels))
    fprintf(fid, 'l|');
end
for i=1:width
    fprintf(fid, '%c|', alignment);
end
fprintf(fid, '}\r\n');

fprintf(fid, '\\hline\r\n');

if(~isempty(colLabels))
    if(~isempty(rowLabels))
        fprintf(fid, '&');
    end
    for w=1:width-1
        fprintf(fid, '\\textit{%s}&', colLabels{w});
    end
    fprintf(fid, '\\textit{%s}\\\\\\hline\r\n', colLabels{width});
end

for h=1:height
    if(~isempty(rowLabels))
        fprintf(fid, '\\textit{%s}&', rowLabels{h});
    end
    for w=1:width-1
        fprintf(fid, '%s  $\\pm $ %s&', matrix1{h, w}, matrix2{h, w});
    end
    fprintf(fid, '%s  $\\pm $ %s\\\\\\hline\r\n', matrix1{h, width}, matrix2{h, width});
end

fprintf(fid, '\\end{tabular}\r\n');

if(~isempty(textsize))
    fprintf(fid, '\\end{%s}', textsize);
end

fclose(fid);
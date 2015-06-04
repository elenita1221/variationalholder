% SIMPLE_ARGS_DEF - Basic parameter management

if ~exist('varargin','var')
    error('The variable ''varargin'' is not defined. Please include it is the arguments of the function.')
end

%%% PARAMETERS MANAGEMENT
for i=1:2:length(varargin)
    if ~isstr(varargin{i}) | ~exist(varargin{i},'var')
        if ~isstr(varargin{i})
            error('invalid parameter');
        else
            switch(varargin{i})
            case 'help'
                clear varargin i
                disp('available options are:')
                disp(who)
                error(' ');
        otherwise
            error(['invalid parameter ' varargin{i}]);
        end
        end
    end
    eval([varargin{i} '= varargin{i+1};']);
end
%%%



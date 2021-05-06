function [Safer_prediction]= Safer(candidate_prediction,baseline_prediction, u)

% SAFEW implements the SAFEW algorithm in [1].
%  ========================================================================
%
%  Input:
%  SAFER takes 3 input parameters in this order:
%
%  candidate_prediction: a matrix with size instance_num * candidate_num . Each column
%                        vector of candidate_prediction is a candidate regression result. 
%                        
%  baseline_prediction: a column vector with length instance_num. It is the regression result
%                       of the baseline method.
%
%  u: the number of unlabeled instances.
%
%  ========================================================================
%
%  Output:
%  Safe_prediction: a predictive regression result by SAFEW.
%
%  ========================================================================
%
%  Example:
%    f = [f1 f2, u];
%    [Safe_prediction]=SAFEW(f,f0,u);
%
%  ========================================================================
%
%  Reference:
%  [1] [1] Lan-Zhe Guo Yu-Feng Li. A General Formulation for Safely Exploiting Weak-Supervision Data In: The 32st AAAI Conference on Artificial Intelligence % (AAAI'18), New Orelans 2018.
%

        m = size(candidate_prediction,2);
        n = size(candidate_prediction,1);
        
        C = candidate_prediction' * candidate_prediction;
        deta = 0.5;
        cvx_begin
        variable z(n,1) nonnegative
        variable epsi(n,1) 
        variable w(n, 1) nonnegative
        variable sln(m, 1) nonnegative
        variable y(n,1)
        minimize (ones(1,n) * epsi + norm(candidate_prediction * sln, 1))
        subject to
            ones(1, m) * sln == 1
            baseline_prediction .* (candidate_prediction * sln) >= ones(n, 1) - epsi
            C * sln >= ones(m, 1) * deta
        cvx_end
        
        Safer_prediction = 0;
        for i = 1:m
            Safer_prediction = Safer_prediction + sln(i)*candidate_prediction(:,i);
        end
        for i = 1 : n
            if Safer_prediction(i) <= 0
                Safer_prediction(i) = -1
            end
        end
        
end

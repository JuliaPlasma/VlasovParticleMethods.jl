
function MaxwellianDistribution(v)
    return 1/(2π) * exp(- dot(v,v) / 2 )
end

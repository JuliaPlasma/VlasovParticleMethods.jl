
abstract type Cache{T} end

_cachehash(T) = hash(Threads.threadid(), hash(T))

struct CacheDict{parentType}
    parent::parentType
    caches::Dict{UInt64, Cache}

    function CacheDict(p)
        caches = Dict{UInt64, Cache}()
        caches[_cachehash(eltype(p))] = p
        new{typeof(p)}(p, caches)
    end
end

Base.parent(cd::CacheDict) = cd.parent

@inline function Base.getindex(c::CacheDict, T::DataType)
    key = _cachehash(T)
    if haskey(c.caches, key)
        c.caches[key]
    else
        c.caches[key] = Cache(T, parent(c))
    end::CacheType(T, parent(c))
end

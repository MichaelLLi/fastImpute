addprocs(5)

## TODO: Implement Breakdown of Feature Matrix when the feature matrix is orthogonal, so that the dependence on p is linear

@everywhere using DataFrames,StatsBase
function sendto(p::Int; args...)
    for (nm, val) in args
        @spawnat(p, eval(Main, Expr(:(=), nm, val)))
    end
end


function sendto(ps::Vector{Int}; args...)
    for p in ps
        sendto(p; args...)
    end
end


function MatrixOptHyperJ(A,B,k,γ,stepsize,ws=true,returnA=true,W=false,debug=true, identity=false)
    if identity
        chunk=1000
        n=size(A)[1]
        p=size(A)[2]
        M=sum(isnan.(A))/(n*p)
        if p != size(B)[1]
            error("Sizes of A and B must match")
        end
        f=size(B)[2]
        startchunk=collect(1:chunk:p)
        if returnA
            Aopt=zeros(Float32,n,p)
            zopt=zeros(Float32,k,f)
            t=0
            for sc in startchunk
                AoptT, zoptT, tT = MatrixOptHyper(A[:,sc:min(sc+chunk-1,p)],B[sc:min(sc+chunk-1,p),sc:min(sc+chunk-1,p)],k,γ,stepsize,ws,returnA,W,debug,identity)
                Aopt[:,sc:min(sc+chunk-1,p)]=AoptT
                zopt[:,sc:min(sc+chunk-1,p)]=zoptT
                t = t + tT
            end
            return Aopt, zopt, t
        else
            zopt=zeros(Float32,k,f)
            for sc in startchunk
                zoptT = MatrixOptHyper(A[:,sc:min(sc+chunk-1,p)],B[sc:min(sc+chunk-1,p),sc:min(sc+chunk-1,p)],k,γ,stepsize,ws,returnA,W,debug,identity)
                zopt[:,sc:min(sc+chunk-1,p)]=zoptT
            end
            return zopt
        end

    else
        if returnA
            Aopt, zopt, t = MatrixOptHyper(A,B,k,γ,stepsize,ws,returnA,W,debug)
            return Aopt, zopt, t
        else
            zopt = MatrixOptHyper(A,B,k,γ,stepsize,ws,returnA,W,debug)
            return zopt
        end

    end
end

function MatrixOptHyper(A,B,k,γ,stepsize,ws=true,returnA=true,W=false,debug=true,identity=false)
# Unknown values should be in NaN for A
# A is the matrix that needs to be imputed, size n x p
# B is the feature selector, size p x f
# k is the maximum rank of the resultant matrix
# ws is warmstart
# returnA is if the algorithm should return the final imputed matrix. If not, only the sparsity variables are returned.
# W is for internal use, keep at false
    zerocolumns=(sum(B.*B,1).>0)[1,:]
    B=B[:,(sum(B.*B,1).>0)[1,:]]
    B=B./sqrt.(sum(B.*B,1))
    n=size(A)[1]
    p=size(A)[2]
    M=sum(isnan.(A))/(n*p)
    if p != size(B)[1]
        error("Sizes of A and B must match")
    end
    f=size(B)[2]
    Ip=eye(p)
    Aopt=zeros(n,p)
    if W==false
        W=Array{Float64}(n,p)
        for i=1:n
            W[i,:]=ones(p)-convert(Vector{Float64},isnan.(A[i,:]))
        end
    end
    println("Preprocessing Complete")
    if ws
        A2=copy(A)
        A2
        output=svds(A,nsv=5)
        z0=output[1][:U]*diagm(output[1][:S])*output[1][:Vt]
    else
        z0=rand(k,f)
        z0=z0/vecnorm(z0)
    end
    A[isnan.(A)]=0
    objbest=Inf
    ∇obj=zeros(k,f)
    i=0
    j=0
    if identity
        obj0, ∇obj0 = Cutting_plane2(A,B,W,z0,k,γ,M,j)
    else
        obj0, ∇obj0 = Cutting_plane(A,B,W,z0,k,γ,M,j)
    end
    noimprov=0
    raised=false
    while i<50
        if (noimprov>=5) && (!raised)
            j=j+1
            raised=true
            objbest=obj0
            noimprov=0
        end
        ∇obj=∇obj0+i/(i+3)*∇obj
        # ∇obj=0.9*∇obj+∇obj0
        ∇tan=-∇obj+dot(∇obj,z0)*z0
        updategrad=∇tan/vecnorm(∇tan)
        z0=z0*cos(pi/stepsize)+updategrad*sin(pi/stepsize)
        z0=z0/vecnorm(z0)
        if identity
            obj0, ∇obj0 = Cutting_plane2(A,B,W,z0,k,γ,M,j)
        else
            obj0, ∇obj0 = Cutting_plane(A,B,W,z0,k,γ,M,j)
        end
        if obj0>objbest
            noimprov=noimprov+1
            if (noimprov % 5==0)&& (raised)
                raised=false
            end
        else
            noimprov=0
            objbest=obj0
            raised=false
        end
        println("There are $noimprov no improvement iterations")
        println("Estimated objective is: $obj0")
        if debug
            realloss=evaluateA(A,B,W,z0)
            println("Real objective is: $realloss")
        end
        i=i+1
    end
    zopt=z0
    X=B*zopt'
    sendto(workers(),X=X)
    println("Model Solved")
    @everywhere function PopulateA(Xtemp,Atemp)
            return X*(pinv(Xtemp'*Xtemp,1e-7)*(Xtemp'*Atemp))
        end
    if returnA
        Xtemp=Array{Float64}[X[W[i,:].==1,:] for i=1:n]
        Atemp=Array{Float64}[A[i,W[i,:].==1] for i=1:n]
        result=pmap(PopulateA,Xtemp,Atemp)
        for i=1:n
            Aopt[i,:]=result[i]
        end
        zoptfull=zeros(k,length(zerocolumns))
        zoptfull[1:k,zerocolumns]=zopt
        return Aopt, zoptfull,obj0
    else
        zoptfull=zeros(k,length(zerocolumns))
        zoptfull[1:k,zerocolumns]=zopt
        return zoptfull
    end

end

# @everywhere using RCall
# @everywhere function Rregression(X,Y)
# @rput X
# @rput Y
# R"""
# data=data.frame(X,Y=Y)
# lm1<-lm(Y~.-1,data=data)
# outcome=lm1$coefficients
# """
# @rget outcome
# outcome[isna.(outcome)]=0
# return outcome
# end
function evaluateA(A,B,W,z0)
    X=B*z0'
    sendto(workers(),X=X)
    @everywhere function PopulateATest(Xtemp,Atemp)
            return X*inv(Xtemp'*Xtemp)*Xtemp'*Atemp
        end
    n=size(A)[1]
    p=size(A)[2]
    Xtemp=Array{Float64}[X[W[i,:].==1,:] for i=1:n]
    Atemp=Array{Float64}[A[i,W[i,:].==1] for i=1:n]
    Aopt=SharedArray{Float64,2}(n,p)
    result=pmap(PopulateATest,Xtemp,Atemp)
    for i=1:n
        Aopt[i,:]=result[i]
    end
    answer=sqrt(mean(((Aopt-A).*(Aopt-A))[A.!=0]))
    return answer
end


function warmstart(A,B,W,k,γ,ws,M)
    num_try = 0
    n=size(A)[1]
    p=size(A)[2]
    f=size(B)[2]
    nsquare=sqrt(n*p)
    if ((p<=100) && (n<=1000)) || ws==false
        z=eye(k,f)
        return z
    else
        chosenp=sample(1:p,min(500,p),replace=false)
        chosenn=sample(1:n,min(max(1000,Int(round(nsquare*log(nsquare)*k/(4*min(500,p)*(1-M)),0))),n),replace=false)
        ncount=length(chosenn)
        pcount=length(chosenp)
        println("warmstarting with $ncount n")
        println("warmstarting with $pcount p")
        Asmall=A[chosenn,chosenp]
        Bsmall=B[chosenp,:]
        Wsmall=W[chosenn,chosenp]
        zopt=MatrixOptHyper(Asmall,Bsmall,k,γ,false,false,Wsmall)
        return zopt
    end
end

# function warmstart2(A,B,W,k,γ)
# @rput A
# R"""
# library(softImpute)
# output<-softImpute(A,lambda=1)
# Apred=output$u%*%diag(output$d)%*%t(output$v)
# """
# @rget Apred
#
# end

function Cutting_plane(A,B,W,z0,k,γ,M,j)
    f=size(B)[2]
    n=size(A)[1]
    p=convert(Int,size(A)[2])
    ∇obj=zeros(k,f)
    nsquare=sqrt(n*p)
    # pnew=p
    # nnew=n
    pnew=min(4*f,p)
    nnew=min(max(100,Int(round(nsquare*log(nsquare)*k*2^j/(8*pnew*(1-M)),0))),n)
    # pnew=min(500,p)
    # nnew=min(max(100,Int(round(nsquare*log(nsquare)*k/(4*min(500,p)*(1-M)),0))),n)
    samplen=sample(1:n,nnew,replace=false)
    samplep=sample(1:p,pnew,replace=false)
    X=B[samplep,:]*z0'
    obj=0
    function SmallInv(Xrow, Wrow, Arow)
        return (eye(Int(sum(Wrow)))- Xrow*inv(eye(k)/γ+Xrow'*Xrow)*Xrow')*Arow
    end
    Wpar=Array{Float64}[W[samplen[i],samplep] for i=1:nnew]
    Xpar=Array{Float64}[X[Wpar[i].==1,:] for i=1:nnew]
    Apar=Array{Float64}[A[samplen[i],samplep[Wpar[i].==1]] for i=1:nnew]
    objpar=map(SmallInv,Xpar,Wpar,Apar)
    obj=@parallel (+) for i=1:nnew
        dot(Apar[i],objpar[i])/(2*pnew*nnew)
    end
    println("Starting DerivCalc")
    println("We chose $nnew n")
    println("We chose $pnew p")
    Btemp1=Array{Float64}(f,nnew)
    Btemp2=Array{Float64}(k,nnew)
    for i=1:nnew
        Btemp1[:,i]=B[samplep[Wpar[i].==1],:]'*objpar[i]
        Btemp2[:,i]=X[Wpar[i].==1,:]'*objpar[i]
    end
    ∇obj=@parallel (+) for i=1:nnew
        -2*γ*Btemp2[:,i]*Btemp1[:,i]'/(2*pnew*nnew)
    end
    # function DerivCalc(index)
    #     return sum(z0[index[1],i]*(B[samplep,index[2]]*B[samplep,i]') for i=1:f)+sum(z0[index[1],i]*(B[samplep,i]*B[samplep,index[2]]') for i=1:f)
    # end
    # Bindex=collect(product(1:k,1:f))
    # println("Starting DerivCalc")
    # Kpar=pmap(DerivCalc,Bindex)
    # println("DerivCalc Done")
    # for l=1:k
    #     for m=1:f
    #         ∇obj2[l,m]=@parallel (+) for i=1:nnew
    #             -γ*(objpar[i]'*diagm(Wpar[i])*Kpar[l+(m-1)*k]*diagm(Wpar[i])*objpar[i])/(2*pnew*nnew)
    #         end
    #         println("l: $l, m: $m")
    #     end
    # end
    # dif=sum(abs(∇obj-∇obj2))
    # println("Difference is $dif")
    println("Derivative Calculated")
    return obj, ∇obj
end


function Cutting_plane2(A,B,W,z0,k,γ,M,j)
    f=size(B)[2]
    n=size(A)[1]
    p=convert(Int,size(A)[2])
    nsquare=sqrt(n*p)
    # pnew=p
    # nnew=n
    pnew=min(2*f,p)
    nnew=min(max(100,Int(round(nsquare*log(nsquare)*k*2^j/(8*pnew*(1-M)),0))),n)
    normfactor=2*pnew*nnew
    ∇obj=zeros(k,f)
    # pnew=min(500,p)
    # nnew=min(max(100,Int(round(nsquare*log(nsquare)*k/(4*min(500,p)*(1-M)),0))),n)
    samplen=sample(1:n,nnew,replace=false)
    samplep=sample(1:p,pnew,replace=false)
    obj=0
    function SmallInv(Xrow, Wrow, Arow)
        return (eye(Int(sum(Wrow)))- Xrow*inv(eye(k)/γ+Xrow'*Xrow)*Xrow')*Arow
    end
    Wpar=Array{Float64}[W[samplen[i],samplep] for i=1:nnew]
    Xpar=Array{Float64}[z0[:,samplep[Wpar[i].==1]]' for i=1:nnew]
    Apar=Array{Float64}[A[samplen[i],samplep[Wpar[i].==1]] for i=1:nnew]
    objpar=map(SmallInv,Xpar,Wpar,Apar)
    obj=@parallel (+) for i=1:nnew
        dot(Apar[i],objpar[i])/normfactor
    end
    println("Starting DerivCalc")
    println("We chose $nnew n")
    println("We chose $pnew p")
    Btemp2=Array{Float64}(k,nnew)
    for i=1:nnew
        Btemp2[:,i]=Xpar[i]'*objpar[i]
        ∇obj[:,samplep[Wpar[i].==1]]+=-2*γ*Btemp2[:,i]*objpar[i]'/normfactor
    end
    # function DerivCalc(index)
    #     return sum(z0[index[1],i]*(B[samplep,index[2]]*B[samplep,i]') for i=1:f)+sum(z0[index[1],i]*(B[samplep,i]*B[samplep,index[2]]') for i=1:f)
    # end
    # Bindex=collect(product(1:k,1:f))
    # println("Starting DerivCalc")
    # Kpar=pmap(DerivCalc,Bindex)
    # println("DerivCalc Done")
    # for l=1:k
    #     for m=1:f
    #         ∇obj2[l,m]=@parallel (+) for i=1:nnew
    #             -γ*(objpar[i]'*diagm(Wpar[i])*Kpar[l+(m-1)*k]*diagm(Wpar[i])*objpar[i])/(2*pnew*nnew)
    #         end
    #         println("l: $l, m: $m")
    #     end
    # end
    # dif=sum(abs(∇obj-∇obj2))
    # println("Difference is $dif")
    println("Derivative Calculated")
    return obj, ∇obj
end

# inverse regularization parameter , usually leave at default
gamma=1000000
# inverse gradient step size, usually leave at defauit
stepsz=64



# Following is a sample - missing values should be labeled NaN

n=10^4
m=10^3
p=1000
k=5
U=rand(Float32,n,k)

R=rand(k,p)
B=rand(m,p)
V=R*B'
# B=eye(Int32,m,p)
# V=rand(Float32,k,m)
Afull=U*V
A=copy(Afull)
A[rand(Float32,n,m).>0.8]=NaN
W=false
tic()
Aopt,zopt,t=MatrixOptHyperJ(A,B,k,gamma,stepsz,false,true,false,false,false)
toc()

sum(abs(Aopt-Afull)./abs(Afull))/(n*m)

# Following is a sample - missing values should be labeled NaN
A=readtable("Netflix/Netflix_1e+05_1000_train.csv")
[A[nm]=convert(DataArray{Float64},A[nm]) for nm in names(A)]
[A[isna.(A[nm]),nm]=NaN for nm in names(A)]
A=convert(Array{Float64},A)
A=A[:,2:end]

B=readtable("Netflix/Netflix_1e+05_1000_features.csv")
B[:intercept]=1
B=convert(Array{Float64},B)
#
# B=eye(size(A)[2])
k=6
tic()
Aopt,zopt,t=MatrixOptHyperJ(A,B,k,gamma,stepsz,false,true,false,false,false)
toc()

Atest=readtable("Netflix/Netflix_1e+05_1000_test.csv")
peopleid=readtable("Netflix/Netflix_1e+05_1000_train.csv")[:,1]
movieid=convert(Array{Int64},readtable("Netflix/Netflix_1e+05_1000_train.csv",header=false)[1,2:end])
error=0
j=0
Aopt[Aopt.>5]=5
Aopt[Aopt.<1]=1
for i=1:size(Atest)[1]
    rownum=find(peopleid.==Atest[:Cust_Id][i])
    colnum=find(movieid.==Atest[:Movie_Id][i])
    if i % 1000 ==0
        print(i)
    end
    if !isempty(rownum)
        error+=(abs(Aopt[rownum,colnum]-Atest[:Rating][i])/Atest[:Rating][i])[1,1]
        j=j+1
    end
end
error=error/j

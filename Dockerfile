
FROM ubuntu:latest
RUN apt-get update && apt-get -y update && apt-get install -y gnupg2

# Set the timezone
ENV TZ=Pacific/Auckland
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Setup python
RUN apt-get install -y build-essential python3.9 python3-pip python3-dev r-base libgeos-dev libproj-dev libcurl4-openssl-dev libssl-dev

# Setup R
# RUN apt update -qq
# RUN apt-get install -y gpg-agent
# RUN apt install --no-install-recommends -y software-properties-common dirmngr
# RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys E298A3A825C0D65DFD57CBB651716619E084DAB9
# RUN add-apt-repository "deb https://cloud.r-project.org/bin/linux/ubuntu $(lsb_release -cs)-cran40/"
# RUN apt install -y --no-install-recommends r-base

# R install
RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys 51716619E084DAB9
RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys E084DAB9
RUN echo "deb http://cran.rstudio.com/bin/linux/ubuntu trusty/" >> /etc/apt/sources.list
RUN cat /etc/apt/sources.list
RUN apt-get update && apt-get install -y --no-install-recommends r-base r-base-dev && apt-get clean

# Utilities for R Jupyter Kernel
RUN echo 'install.packages(c("base64enc","evaluate","IRdisplay","jsonlite","uuid","digest"), \
repos="http://cran.us.r-project.org", dependencies=TRUE)' > /tmp/packages.R \
   && Rscript /tmp/packages.R

# Install R Jupyter Kernel
RUN echo 'install.packages(c("repr", "IRdisplay", "crayon", "pbdZMQ", "devtools"),repos="http://cran.us.r-project.org", dependencies=TRUE)' > /tmp/packages.R && Rscript /tmp/packages.R
RUN echo 'install.packages(c("IRkernel"),repos="http://cran.us.r-project.org", dependencies=TRUE)' > /tmp/packages.R && Rscript /tmp/packages.R

RUN apt-get install -y build-essential libcurl4-openssl-dev libssl-dev libxml2
RUN apt-get install -y libxml2-dev
RUN apt install -y graphviz
RUN apt-get install -y git
RUN pip3 -q install pip --upgrade

# Create working directory
RUN mkdir src
WORKDIR src/
COPY . .

# Install python packages
RUN pip3 install --upgrade cython numpy pyproj sklearn graphviz xarray
RUN pip3 install jupyter pandas matplotlib rpy2 fiona cartopy geopandas shapely
RUN pip3 install --index-url https://support.bayesfusion.com/pysmile-B/ pysmile


# Set working directory
WORKDIR /src/

ENV TINI_VERSION v0.6.0
ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /usr/bin/tini
RUN chmod +x /usr/bin/tini
ENTRYPOINT ["/usr/bin/tini", "--"]

CMD ["jupyter", "notebook", "--port=8888", "--no-browser", "--ip=0.0.0.0", "--allow-root"]

RUN echo 'IRkernel::installspec()' > /tmp/temp.R && Rscript /tmp/temp.R
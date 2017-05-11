#!/bin/sh
# Disable SELinux
setenforce 0

# Download Ambari Repository
wget http://public-repo-1.hortonworks.com/ambari/centos7/2.x/updates/2.5.0.3/ambari.repo -O /etc/yum.repos.d/ambari.repo

# Install java-1.8
yum -y install java-1.8.0-openjdk java-1.8.0-openjdk-devel

# Install ambari-server
yum -y install ambari-server

# bootstrap ambari-server
ambari-server setup -s --java-home=/usr/lib/jvm/jre/

# start ambari-server
ambari-server start

sh install_ambari_agent.sh

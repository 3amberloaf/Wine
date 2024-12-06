#!/bin/bash

# Dynamically generate core-site.xml
cat > $HADOOP_CONF_DIR/core-site.xml <<EOF
<configuration>
    <property>
        <name>fs.s3a.access.key</name>
        <value>${AWS_ACCESS_KEY_ID}</value>
    </property>
    <property>
        <name>fs.s3a.secret.key</name>
        <value>${AWS_SECRET_ACCESS_KEY}</value>
    </property>
    <property>
        <name>fs.s3a.session.token</name>
        <value>${AWS_SESSION_TOKEN}</value>
    </property>
</configuration>
EOF

# Run the default CMD
exec "$@"

{{- define "holmes-mcp.name" -}}
{{- .Chart.Name | trunc 63 | trimSuffix "-" }}
{{- end }}

{{- define "holmes-mcp.fullname" -}}
{{- if contains .Chart.Name .Release.Name }}
{{- .Release.Name | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-%s" .Release.Name .Chart.Name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}

{{- define "holmes-mcp.labels" -}}
helm.sh/chart: {{ printf "%s-%s" .Chart.Name .Chart.Version }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
app.kubernetes.io/part-of: holmes-mcp
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}

{{- define "holmes-mcp.selectorLabels" -}}
app.kubernetes.io/name: {{ include "holmes-mcp.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{- define "holmes-mcp.redis.fullname" -}}
{{- printf "%s-redis" (include "holmes-mcp.fullname" .) | trunc 63 | trimSuffix "-" }}
{{- end }}

{{- define "holmes-mcp.redis.selectorLabels" -}}
app.kubernetes.io/name: {{ include "holmes-mcp.name" . }}-redis
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{- define "holmes-mcp.redis.secretName" -}}
{{- default (include "holmes-mcp.redis.fullname" .) .Values.redis.existingSecret }}
{{- end }}

{{/* True when the MCP server has a Redis to use. */}}
{{- define "holmes-mcp.redis.configured" -}}
{{- if or .Values.redis.enabled .Values.redis.external.existingSecret }}true{{ end }}
{{- end }}

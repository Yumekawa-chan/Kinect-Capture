Shader "Custom/ColoredVertex" {
	Properties{
	 _Size("ShaderSize", Float) = 1.000000
	 _PointSize("PointSize", Float) = 1.0
	}
		SubShader{
		 Pass {
		CGPROGRAM
		#pragma vertex vert
		#pragma fragment frag
		#pragma target 2.0
		#include "UnityCG.cginc"
		#pragma multi_compile_fog
		#define USING_FOG (defined(FOG_LINEAR) || defined(FOG_EXP) || defined(FOG_EXP2))

		float _Size;
		float _PointSize;

		// uniforms
		// vertex shader input data
		struct appdata {
		  float3 pos : POSITION;
		  half4 color : COLOR;
		  UNITY_VERTEX_INPUT_INSTANCE_ID
		};

	// vertex-to-fragment interpolators
	struct v2f {
	  float psize : PSIZE;
	  fixed4 color : COLOR0;
	  #if USING_FOG
		fixed fog : TEXCOORD0;
	  #endif
	  float4 pos : SV_POSITION;
	  UNITY_VERTEX_OUTPUT_STEREO
	};

	// vertex shader
	v2f vert(appdata IN) {
		IN.pos.xyz = IN.pos.xyz * _Size;
		
	  v2f o;
	  UNITY_SETUP_INSTANCE_ID(IN);
	  UNITY_INITIALIZE_VERTEX_OUTPUT_STEREO(o);
	  half4 color = IN.color;
	  float3 eyePos = mul(UNITY_MATRIX_MV, float4(IN.pos,1)).xyz;
	  half3 viewDir = 0.0;
	  o.color = saturate(color);
	  // compute texture coordinates
	  // fog
	  #if USING_FOG
		float fogCoord = length(eyePos.xyz); // radial fog distance
		UNITY_CALC_FOG_FACTOR_RAW(fogCoord);
		o.fog = saturate(unityFogFactor);
	  #endif
		// transform position
		o.pos = UnityObjectToClipPos(IN.pos);
		o.psize = _PointSize;
		return o;
	  }


	// fragment shader
	fixed4 frag(v2f IN) : SV_Target {
	  fixed4 col;
	  col = IN.color;
	  // fog
	  #if USING_FOG
		col.rgb = lerp(unity_FogColor.rgb, col.rgb, IN.fog);
	  #endif
	  return col;
	}
	ENDCG
	 }
	}
}

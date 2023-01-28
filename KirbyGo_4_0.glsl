// -------------------------------Copyright Declaration------------------------------
// This project is strongly based on Inigo Quilez's project Happing Jumping
// https://www.shadertoy.com/view/3lsSzf

// github: https://github.com/wongzingji/KirbyGo

//---------------------------------------NOTICE--------------------------------------
// To see waterfall texture, you would have to load custom textures using console
// by executing the 2 following lines
// gShaderToy.SetTexture(1, {mSrc:'https://dl.dropboxusercontent.com/s/0s7mrlje6xgl0kr/displacement.png?dl=0', mType:'texture', mID:1, mSampler:{ filter: 'mipmap', wrap: 'repeat', vflip:'true', srgb:'false', internal:'byte' }});
// gShaderToy.SetTexture(0, {mSrc:'https://dl.dropboxusercontent.com/s/bqrco6zi2yl6x5i/uniformclouds.png?dl=0', mType:'texture', mID:1, mSampler:{ filter: 'mipmap', wrap: 'repeat', vflip:'true', srgb:'false', internal:'byte' }});

// AND: waterfall is set at z=-20.0, so as Kirby jumps forward, waterfall does disappear
//---------------------------------------NOTICE--------------------------------------

// #if HW_PERFORMANCE==0
// #define AA 1
// #else
#define AA 1  // Set AA to 1 if your machine is too slow
// #endif


////////////////////////////////
// add elements
// waterfall texture
vec3 wftexel(vec2 uv, float time)
{
    //vec3 bd = vec3(0.66,0.196,0.188);
    //vec3 td = vec3(0.463,0.188,0.165);
    //vec3 bl = vec3(1.0,1.0,1.0);
    //vec3 tl = vec3(0.9,0.757,0.737);
    vec3 bd = vec3(0.35,0.07,0.06);
    vec3 td = vec3(0.121,0.02,0.012);
    vec3 bl = vec3(198.0/255.0,121.0/255.0,121.0/255.0);
    vec3 tl = vec3(0.772,0.376,0.32);
    
    float threshold = 0.55;
    
    vec2 uv2 = uv * vec2(1.6,1.0);

    // displacement
    vec2 displ = texture(iChannel1, uv2 + iTime / 5.0).xy;
    displ = (displ*2.0-1.0) * 0.04;  // displacement amount
    
    // noise
    float noise = texture(iChannel0, vec2( uv.x*2.0, uv.y*0.2 + iTime/5.0 ) + displ).r;
    noise = round(noise*5.0)/5.0;
    
    vec3 col = mix(mix(bd,td,uv.y),mix(bl,tl,uv.x),noise);
    col = mix(vec3(1.0), col, step(threshold,uv.y+displ.y));
    return col;
}

// watersurface
vec3 watersurf(vec2 uv, float time)
{   
    uv *= vec2(3.0,3.0);
    float freq = 4.0;
    float amp = 1.0;
    float n = 0.0;
    vec2 g, gsum = vec2(0.0);

    for (int i=0; i<5; i++) {
        n += amp*psrdnoise(uv*freq+gsum*0.14, vec2(0.0), 8.0/freq*iTime, g);
        gsum += g*amp;
        freq *=2.0;
        amp *= 0.5;
    }

    vec3 mixcolor = mix(vec3(0.463,0.188,0.165), vec3(0.35), -n*0.35+0.5);
    return mixcolor;
}

// mushroom
float head( in vec3 p )
{
    // top
    float d3 = sdEllipsoid( p+vec3(0.0,2.1,0.0), vec3(0.35,0.2,0.35) );
    d3 -= 0.03*(0.5+0.5*sin(11.0*p.z)*cos(9.0*p.x));
    //d3 -= 0.05*exp(-128.0*dot(p.xz,p.xz));
    
    // interior
    float d4 = sdSphere( p+vec3(0.0,2.3,0.0), 0.35 );
	d4 += 0.005*sin( 20.0*atan(p.x,p.z) );

    // substract
    return smax( d3, -d4, 0.02 );
    //return d3;
}

vec4 mapMushroom(vec3 p)
{
    vec2 objXY = vec2(0., 0.);
    // objID: 7.0, 8.0
    // stem
    float h = clamp(p.y+1.2,0.0,1.0);
    vec3 o = 0.12 * sin( h*3.0 + vec3(0.0,2.0,4.0) );
    o = o*4.0*h*(1.0-h) * h;
    
    float c = cos(0.6*p.x+p.y+0.5);
    float s = sin(0.6*p.x+p.y+0.5);
    mat2 rot = mat2(c,s,-s,c);
    vec3 q = p+vec3(0.0,1.9,0.0)-o*vec3(1.0,0.0,1.0);
    q.xz = rot*q.xz;

    float d1 = sdSegment( q, vec3(-0.15,0.0,0.0), vec3(-0.11,1.45,0.00) ).x;
    d1 -= 0.06;
    d1 -= 0.1*exp(-16.0*h);
    vec4 res = vec4(d1, 8.0, objXY);

    // head
    float d2 = head( p + vec3(0.06,-1.5,0.0));
    vec4 res2 = vec4(d2, 7.0, objXY);

    // mix head and stem
    //d1 = smin( d1, d3, 0.2 );
    //d1 *= 0.75; 
    res = opU(res2, res);
        
    return res;
}

// waterfall
vec4 sdWaterfall( in vec3 pos )
{
    vec2 objXY = vec2(0.,0.);
    vec3 q = pos - vec3(0.0,0.6,-10.0);
    float mw = 100.5;  // waterfall width
    float mh = 4.0;
    float hgap = 0.0; // height difference between boxbehind and waterfall
    float mate = 9.0; // set default material number to waterfall

    float d = sdRoundBox(q,vec3(mw,mh,0.3),0.1); // waterfall
    float d2 = sdRoundBox(q-vec3(0.0,-2.0*hgap,-0.2),vec3(mw,mh-hgap,0.2),0.2); // box behind
    
    if(d > d2)
    {
        mate = 12.0;  // boxbehind
    }
    d = min(d,d2);
   
    return vec4(d,mate,objXY);
}

// pond
vec4 sdPond( in vec3 pos )
{
    vec2 objXY = vec2(0.0,0.0);
    vec3 q = pos-vec3(-2.2,-0.05,7.3);
    float w = 0.8; // width/height of pond border
    float h = 0.08;
    // outside box
    float d1 = sdRoundBox(q,vec3(w,h,w),0.0);
    
    // inside box
    float d2 = sdRoundBox(q,vec3(w-0.2,h+0.01,w-0.2),0.0);
    d1 = max(d1,-d2);
    
    // water surface
    q.y += 0.04;
    //float d3
    float d3 = sdRoundBox(q,vec3(w,0.06+0.02*sin(3.0*iTime),w),0.0);
    
    return d1 > d3 ? vec4(d3,10.0,objXY) : vec4(d1,11.0,objXY);
}

vec2 sdFloor(in vec3 pos, float atime)
{
    float floorHeight = -0.1
                                     - 0.05*(sin(pos.x*2.0)+sin(pos.z*2.0));
    float t5 = fract(atime+0.05);
    //float k = length(pos.xz-cen.xz);
    float k = length(pos.xz);
    float t2 = t5*15.0-6.2831 - k*3.0;
    floorHeight -= 0.1*exp(-k*k)*sin(t2)*exp(-max(t2,0.0)/2.0)*smoothstep(0.0,0.01,t5);
    float dFloor = pos.y - floorHeight;

    return vec2(floorHeight,dFloor);
}

float sdCloud(vec3 p)
{	
	p.x -= iTime;		// translate with time
	//p += snoise(p*0.5)*1.0;	// domain warp!
	
	vec3 q = p -vec3(p.x*0.4-p.z*0.5,10.0,p.z*0.5);
	// repeat on grid
	q.xz = mod(q.xz - vec2(2.5), 5.0) - vec2(2.5);
    q.y *= 2.0;	// squash in y
	float d = length(q) - _SphereRadius;	// distance to sphere

	// offset distance with noise
	//p = normalize(p) * _SphereRadius;	// project noise point to sphere surface
	p.y -= iTime*0.3;	// animate noise with time
	d += fbm(p*_NoiseFreq) * _NoiseAmp;
	return d;
}

#define ZERO (min(iFrame,0))
//////////////////////////////// 
//OLD:vec2->NEW:vec4(add relative coordinate)
vec4 map( in vec3 pos, float atime )
{
    //Defalut return value
    vec2 objXY = vec2(0.,0.); 
    vec4 res = vec4(-1.,-1., objXY);
    
    //yMove
    float t = fract(atime); //[0,1]
    float bounceWave = 4.0*t*(1.0-t);
    float yMove = pow(bounceWave,2.0-bounceWave) + 0.1;
    
    //zMove
    float zMove = 0.7*(floor(atime) + pow(t,0.7) -1.0);
        
     //xMove
    float tt = abs(fract(atime*0.5)-0.5)/0.5;
    float xMove = 0.5*(-1.0 + 2.0*tt);
     
    vec3 cen = vec3( xMove, yMove, zMove);
    
    //Coordinate
    vec3 basic = pos-cen; 
    basic.xy = rotMat(-xMove*0.3) * basic.xy;
    
    float ttt = abs(fract((atime+0.5)*0.5)-0.5)/0.5;
    float bodyMove = 0.5*(-1.0 + 2.0*ttt);
    basic.xz = rotMat(bodyMove*1.) * basic.xz;
    
    vec3 symmBody =  vec3(  abs(basic.x), basic.yz);
 
    
    //body
    float sy = 0.9 + 0.1*bounceWave;
    float compress = 1.0-smoothstep(0.0,0.4,bounceWave);
    sy = sy*(1.0-compress) + compress;
    float sx = 1./sy;
    
    float dBody = sdEllipsoid( basic , vec3(0.2, 0.2, 0.2) );

    //arm
    vec3 armPos = vec3(0.2, 0., 0.);
    float dArm = sdSphere( symmBody - armPos, 0.07);
    
    dBody = smin(dBody, dArm, 0.03);
    
    //mouth
    {
    float smell = 23.* basic.x*basic.x;
    vec3 mouthPos = vec3(0., -0.03 +smell, 0.17);
    float dMouth = sdEllipsoid( basic - mouthPos, vec3(0.1, 0.11, 0.15)*0.3 );
    
    dBody = smax(dBody, -dMouth, 0.01);
    
    vec2 frontBody= vec2(-1., -1.);    
    if (basic.z > 0.){frontBody = symmBody.xy;}

    vec4 bodyObj = vec4(dBody, 2.0, frontBody);
    res = bodyObj;
    }
    
    // leg
    {
    float sy = 0.5 + 0.3*bounceWave;
    float compress = 1.0-smoothstep(0.0,0.4,bounceWave);
    sy = sy*(1.0-compress) + compress;
    float sx = 1./sy;
    
    vec3 legPos = vec3(0.11, -0.17, 0.05);
    legPos = symmBody - legPos;
    legPos.y += 0.2* legPos.x*legPos.x;
    
    float t6 = cos(6.2831*(atime*0.5+0.25));
    legPos.xz =rotMat(-1.0) * legPos.xz;
    legPos.xy =rotMat(t6*sign(basic.x)) * legPos.xy;
        
    float dleg = sdEllipsoid( legPos, vec3(0.43, 0.18, 0.28)*0.3 );
    vec4 legObj = vec4(dleg, 3.0,objXY);
    
    res = opU(res, legObj);
    }
    
    
    //eye
    vec3 eyePos = vec3(0.06, 0.05, 0.19-2.*basic.y*basic.y);
    eyePos = symmBody - eyePos;
    
    eyePos.xz = rotMat(0.32)*eyePos.xz;
    
    float ss = min(1.,mod(atime,3.1));
    float sss = 1.- (smoothstep(0.,0.1,ss)-smoothstep(0.18,0.4,ss));
    
    float dEye = sdEllipsoid( eyePos, vec3(0.15, 0.3*sss+0.001, 0.03*sss+0.001)*0.2 );
    vec4 eyeObj = vec4(dEye, 4.0, eyePos.xy);
    res = opU(res, eyeObj);
    
    
    // ground
    vec2 floorData = sdFloor(pos, atime);
    float floorHeight = floorData.x;
    float dFloor = floorData.y;
    vec2 floorObj = vec2(dFloor, 0.0);
    
    // bubbles
    vec3 bubbleArea = vec3( mod(abs(pos.x),3.0),pos.y,mod(pos.z+1.5,3.0)-1.5);
    
    vec2 id = vec2( floor(pos.x/3.0), floor((pos.z+1.5)/3.0) );
    float fid = id.x*11.1 + id.y*31.7;

    float fy = fract(fid*1.312+atime*0.1);
    float y = -1.0+4.0*fy;
    
    vec3  rad = vec3(0.7,1.0+0.5*sin(fid),0.7);
    rad -= 0.1*(sin(pos.x*3.0)+sin(pos.y*4.0)+sin(pos.z*5.0));    
    
    //smoothly  change the size when fly
    float siz = 4.0*fy*(1.0-fy);
    float dTree = sdEllipsoid( bubbleArea-vec3(2.0,y,0.0), rad*siz );

    //float bubbleTexture = 0.2*(-1.0+2.0*smoothstep(-0.2,0.2, sin(18.0*pos.x)+sin(18.0*pos.y)+sin(18.0*pos.z))); 
    //dTree += 0.01 * bubbleTexture;

    
    dTree *= 0.6;
    dTree = min(dTree,2.0);
    vec2 treeObj = vec2(dTree,1.0);

    vec2 res2 = smin( floorObj, treeObj, 0.32 );
    //vec4 floorObj = vec4(dFloor, 1.0, objXY);
    
    res = opU(res, vec4(res2,objXY));
    
    //donuts
    {
    float fs = 5.0;
    vec3 qos = fs*vec3(pos.x, pos.y-floorHeight-0.02, pos.z );
    vec2 id = vec2( floor(qos.x+0.5), floor(qos.z+0.5) );
    
    vec3 vp = vec3( fract(qos.x+0.5)-0.5,qos.y,fract(qos.z+0.5)-0.5);
    vp.xz += 0.1*cos( id.x*130.143 + id.y*120.372 + vec2(0.0,2.0) );
    
    float den = sin(id.x*0.1+sin(id.y*0.091))+sin(id.y*0.1);
    float fid = id.x*0.143 + id.y*0.372;
    float ra = smoothstep(0.0,0.1,den*0.1+fract(fid)-0.9);
    
    if (ra < 0.5)
    {
        ra = -0.1;
    }

    vec4 donutObj;
    vec4 creamObj;

    float angle = -1.2*fid;
    vp.xy = rotMat(angle) * vp.xy;

    float dDonut = sdDonut(vp, 0.24*ra)/fs;
    float dCream = sdCream(vp, 0.24*ra)/fs;
    //float dCandy = sdSphere( vp, 0.35*ra )/fs;
    donutObj = vec4(dDonut, 5.0, objXY);
    creamObj = vec4(dCream, 6.0, objXY);

    res = opU(res, donutObj);
    res = opU(res, creamObj);
    }

    // mushroom
    {
    vec3 q = pos;
    q.xz = mod(q.xz - vec2(2.5), 5.0) - vec2(2.5); // repeat
    q.y -= 1.05;
    vec4 mushroomObj = mapMushroom(q);
    res = opU(res, mushroomObj);
    }

    // waterfall
    {
    vec3 q = pos;
    vec4 wfObj = sdWaterfall(q);
    res = opU(res,wfObj);
    }
    
    // pond
    {
    vec3 q = pos;
    vec4 pondObj = sdPond(q);
    res = opU(res,pondObj);
    }

    return res;
}


vec4 castRay( in vec3 ro, in vec3 rd, float time )
{
    vec4 res = vec4(-1.0,-1.0, 0., 0.);

    float tmin = 0.5;
    float tmax = 20.0;
    
    float t = tmin;
    for( int i=0; i<256 && t<tmax; i++ )
    {
        vec4 h = map( ro+rd*t, time );
        if( abs(h.x)<(0.0005*t) )
        { 
            res = vec4(t,h.yzw); 
            break;
        }
        t += h.x;
    }
    
    return res;
}


vec3 calcNormal( in vec3 pos, float time )
{

    // vec2 e = vec2(0.0005,0.0);
    // return normalize( vec3( 
    //     map( pos + e.xyy, time ).x - map( pos - e.xyy, time ).x,
	// 	map( pos + e.yxy, time ).x - map( pos - e.yxy, time ).x,
	// 	map( pos + e.yyx, time ).x - map( pos - e.yyx, time ).x ) );

    vec3 n = vec3(0.0);
    for( int i=min(iFrame,0); i<4; i++ )
    {
        vec3 e = 0.5773*(2.0*vec3((((i+3)>>1)&1),((i>>1)&1),(i&1))-1.0);
        n += e*map(pos+0.0005*e,time).x;
    }
    return normalize(n);    
}


float calcOcclusion( in vec3 pos, in vec3 nor, float time )
{
	float occ = 0.0;
    float sca = 1.0;
    for( int i=0; i<5; i++ )
    {
        float h = 0.01 + 0.11*float(i)/4.0;
        vec3 opos = pos + h*nor;
        float d = map( opos, time ).x;
        occ += (h-d)*sca;
        sca *= 0.95;
    }
    return clamp( 1.0 - 2.0*occ, 0.0, 1.0 );
}

vec3 calcTransmittance(vec3 ro, vec3 rd, float tmin, float tmax, float atten, float time)
{
    const int MAX_DEPTH = 4;
    float hitPoints[MAX_DEPTH];
    int depth = 0;
    
    for (float t = tmin; t < tmax;)
    {
        float h = abs(map(ro + t * rd, time).x);
        if (h < 1e-5) { hitPoints[depth++] = t; t += 0.01; };
        if (depth >= MAX_DEPTH) break;
        t += h;
    }
    
    float thickness = 0.0;
    for (int i = 0; i < depth - 1; i += 2) thickness += hitPoints[i+1] - hitPoints[i];
    
    return vec3(1.0) * exp(-atten * thickness * thickness);
}

// maps position to color
vec4 calcVolume(vec3 p)
{
	float d = sdCloud(p);
	vec4 c = mix(innerColor, outerColor, smoothstep(0.5, 1.0, d));
	c.rgb *= smoothstep(-1.0, 0.0, p.y)*0.5+0.5;	// fake shadows
	float r = length(p)*0.04;
	c.a *= exp(-r*r);	// fog
	return c;
}

float sampleLight(vec3 pos, vec3 sun_lig)
{
	const int _LightSteps = 8;
	const float _ShadowDensity = 1.0;
	vec3 lightStep = (sun_lig * 2.0) / float(_LightSteps);
	float t = 1.0;	// transmittance
	for(int i=0; i<_LightSteps; i++) {
		vec4 col = calcVolume(pos);
		t *= max(0.0, 1.0 - col.a * _ShadowDensity);
		//if (t < 0.01)
			//break;
		pos += lightStep;
	}
	return t;
}

vec4 castRayVolume(vec3 rayOrigin, vec3 rayStep, vec4 sum, out vec3 pos)
{
	pos = rayOrigin;
	for(int i=0; i<_VolumeSteps; i++) {
		vec4 col = calcVolume(pos);
#if 0
		// volume shadows
		if (col.a > 0.0) {
			col.rgb *= sampleLight(pos, normalize( vec3(0.6, 0.35, 0.5) )); // sun_dir	
		}
#endif		
		
#if 0
		sum = mix(sum, col, col.a);	// under operator for back-to-front
#else	
		col.rgb *= col.a;		// pre-multiply alpha
		sum = sum + col*(1.0 - sum.a);	// over operator for front-to-back
#endif
		
#if 0
		// exit early if opaque
        	if (sum.a > _OpacityThreshold)
            		break;
#endif		
		pos += rayStep;
		//rayStep *= 1.01;
	}
	return sum;
}

vec3 render( in vec3 ro, in vec3 rd, float time, vec2 uv )
{ 
    // sky dome
    vec3 col = vec3(0.5, 0.8, 0.9) - max(rd.y,0.0)*0.5;
    vec3 hitPos;
    col += castRayVolume(ro, rd*0.7, vec4(0.0), hitPos).rgb; // rd*stepSize

    
    vec4 res = castRay(ro,rd, time);
    if( res.y>-0.5 )
    {
        float t = res.x;
        vec2 objXY = res.zw;
        vec3 pos = ro + t*rd;
        vec3 nor = calcNormal( pos, time );
        vec3 ref = reflect( rd, nor );
        
		col = vec3(0.2);
        float ks = 1.0;

        if (res.y > 11.5)  // box behind waterfall 12
        {
            col = vec3(0.66,0.196,0.188);
        }
        else if (res.y > 10.5) // pond border 11
        {
            //vec3 bd = vec3(0.66,0.196,0.188);
            //vec3 td = vec3(0.463,0.188,0.165);
            //vec3 bl = vec3(1.0,1.0,1.0);
            //vec3 tl = vec3(0.9,0.757,0.737);
            col = vec3(0.463,0.188,0.165);
        }
        else if (res.y > 9.5) // water surface 10
        {
            col = watersurf(uv, time);
        }
        else if (res.y > 8.5) // waterfall 9
        {
            col = wftexel(uv, time);
        }
        else if (res.y > 7.5) // mushroom stem
        {
            col = vec3(0.6706, 0.2, 0.0549);   
            col = mix( col, 0.6*vec3(0.2078, 0.098, 0.1059), 0.92*(1.0-smoothstep(0.1,0.5,pos.y)) );
        }
        else if (res.y > 6.5)
        {
            col = vec3(0.76,0.26,0.3); 
             vec2 id = floor(5.0*pos.xz+0.5);
		     col += 0.036*cos((id.x*11.1+id.y*37.341) + vec3(0.0,1.0,2.0) );
             col *= 3.0*noise(pos.xy);

             vec2 q = vec2(0.);
            q.x = fbm( pos.xy);
            q.y = fbm( pos.xy + vec2(1.0));

            vec2 r = vec2(0.);
            r.x = fbm( pos.xy + 1.0*q + vec2(1.7,9.2));
            r.y = fbm( pos.xy + 1.0*q + vec2(8.3,2.8));

            float f = fbm(pos.xy+r);
            vec2 g = vec2(f);
            
            vec3 color = vec3(0.0);
            color = mix(vec3(0.681,0.858,0.920),
                        vec3(0.967,0.156,0.573),
                        clamp((f*f)*4.312,0.992,1.0));

            color = mix(color,
                        vec3(0.300,0.034,0.134),
                        clamp(length(q),0.0,1.0));

            color = mix(color,
                        vec3(1.000,0.700,0.315),
                        clamp(length(r.x),0.0,1.0));
            
            col *= vec3((f*f*f+0.7*f*f*f*f+3.068*f*f)*color*5.0);
        }
        else if (res.y > 5.5)  //donuts
        {
            col = vec3(0.14,0.048,0.0); 
             vec2 id = floor(5.0*pos.xz+0.5);
		     col += 0.13*cos((id.x*11.1+id.y*37.341) + vec3(1.0,1.0,1.0) );
             col = max(col,0.0);
            //col = vec3(0.639,0.302,0.549);
          
        }
        else if (res.y > 4.5)  //candy
        {
            col = vec3(0.14,0.048,0.0); 
             vec2 id = floor(5.0*pos.xz+0.5);
		     col += 0.036*cos((id.x*11.1+id.y*37.341) + vec3(0.0,1.0,2.0) );
             col = max(col,0.0);
        }
        else if( res.y>3.5 ) // eye
        {  // todo: return relative coordinate instead of absolute
            vec3 black = vec3(0.0);
            vec3 blue = vec3(0.0,0.0,1.0);
            col = (1.0 - vec3(smoothstep(-0.2,0.,objXY.y))) * blue;
            
            vec2 ab = vec2(0.0,0.02);
             float eyeBall = pow(objXY.x-ab.x,2.0)/1. + pow(objXY.y-ab.y,2.0)/4.;
             if (eyeBall < 0.0002){col = vec3(1.0);}
            
        } 
        else if( res.y>2.5 ) // leg
        { 
            col = vec3(0.4,0.0,0.02);
        } 
        else if( res.y>1.5 ) // body
        { 
            col = vec3(0.5,0.07,0.1);
            if (objXY.x > -0.5){
            
                vec3 bodyCol = vec3(0.5,0.07,0.1);
                vec3 blusherCol = vec3(0.686,0.031,0.067);
                vec3 mouthCol = vec3(0.561,0.020,0.047);

                vec2 ab = vec2(0.12,-0.02); 
                float blusher = pow(objXY.x-ab.x,2.0)/4. + pow(objXY.y-ab.y,2.0)/1.;
                col = mix(blusherCol, bodyCol, smoothstep(0.0001, 0.0002, blusher));
                
                float mouth = pow(objXY.x,2.0)/4. + pow(objXY.y-12.*pow(objXY.x, 2.0)+0.04,2.0)/4.;
                if (mouth < 0.00015){col = mouthCol;}
               }
        } else if (res.y>0.5) // bubble
        {
            col = bubble_mat.diffuseAlbedo;
        } 
        else // ground
        {
            col = vec3(0.3961, 0.2118, 0.2196);
            
            float wave = sqrt(1.0-pow(fract(pos.x+pos.y*2.0+pos.z*0.5), 2.0)) + pow(pos.x*pos.x, 0.33);
            float f = 0.2*(-1.0+2.0*smoothstep(-0.2,0.2,wave));
            col += 0.6*wave*vec3(0.1529, 0.0902, 0.1098);
            col *= vec3(0.6392, 0.4431, 0.4431);            // vec3(0.06,0.06,0.02)
            
            // stronger light
            ks = 0.5 + pos.y*0.15;
        }
        
        // lighting (sun, sky, bounce)
        vec3  sun_lig = normalize( vec3(0.6, 0.35, 0.5) );
        float sun_dif = clamp(dot( nor, sun_lig ), 0.0, 1.0 );
        vec3  sun_hal = normalize( sun_lig-rd );
        float sun_sha = step(castRay( pos+0.001*nor, sun_lig,time ).y,0.0);
		float sun_spe = ks*pow(clamp(dot(nor,sun_hal),0.0,1.0),8.0)*sun_dif*(0.04+0.96*pow(clamp(1.0+dot(sun_hal,rd),0.0,1.0),5.0));
		float sky_dif = sqrt(clamp( 0.5+0.5*nor.y, 0.0, 1.0 ));
        float bou_dif = sqrt(clamp( 0.1-0.9*nor.y, 0.0, 1.0 ))*clamp(1.0-0.1*pos.y,0.0,1.0);

		vec3 lin = vec3(0.0);
        lin += sun_dif*vec3(8.10,6.00,4.20)*sun_sha;
        lin += sky_dif*vec3(0.50,0.70,1.00);
        lin += bou_dif*vec3(0.40,1.00,0.40);
        
        if (res.y>0.5 && res.y<1.5)
        {
            // translucency
            float t = clamp(0.5, 0.2, 1.0);
            lin *= t*calcTransmittance(pos+nor*vec3(0.01), sun_lig, 0.01, 10.0, 2.0, time);
            lin += (1.0 - t) * calcTransmittance(pos+nor*vec3(0.01), rd, 0.01, 10.0, 0.5, time);
        }

        //col = col*lin;
        
        if (sun_dif < 0.256) {
            col *= 0.25; // 0.195
        } else if (sun_dif < 0.59) {
            col *= 0.55;
        } else if (sun_dif < 0.781) {
            col *= 0.781;
        } else {
            col *= 0.900;
        }
        
		col += sun_spe*vec3(8.10,6.00,4.20)*sun_sha;
        if (res.y>0.5 && res.y<1.5)
        {
            col += lin * bubble_mat.specularAlbedo * pow(max(0.0, dot(reflect(sun_lig,nor),rd)), 4.0);
            col = mix( col, vec3(0.3961, 0.2118, 0.2196), 0.92*(1.0-smoothstep(0.1,0.5,pos.y)) );
        }

        col = mix( col, vec3(0.5,0.7,0.9), 1.0-exp( -0.0001*t*t*t ) );
    }

    return col;
}

void mainImage( out vec4 fragColor, in vec2 fragCoord )
{
    vec3 tot = vec3(0.0);
#if AA>1
    for( int m=ZERO; m<AA; m++ )
    for( int n=ZERO; n<AA; n++ )
    {
        vec2 o = vec2(float(m),float(n)) / float(AA) - 0.5;
        vec2 p = (-iResolution.xy + 2.0*(fragCoord+o))/iResolution.y;
        vec2 uv = (fragCoord+o)/min(iResolution.x, iResolution.y);
        float d = 0.5+0.5*sin(fragCoord.x*147.0)*sin(fragCoord.y*131.0);
        float time = iTime - 0.5*(1.0/24.0)*(float(m*AA+n)+d)/float(AA*AA);;
#else
        vec2 p = (-iResolution.xy + 2.0*fragCoord)/iResolution.y;
        vec2 uv = fragCoord/min(iResolution.x, iResolution.y);
        float time = iTime;
#endif
        //float time = 1.;
        time *= 0.7;

        // camera	  
        float forwardWave = sin(0.5*time);
        float smoothForward = 0.7*(-1.+time*1.0 - 0.4*forwardWave); 
        
        
        ////////////////////////****V2.0***********************///////////////////
        //float rotx = 2.5 + (iMouse.y / iResolution.y)*4.0;
        //float roty = 2.5 + (iMouse.x / iResolution.x)*4.0;
        //vec3  ta = vec3( 0.0, 0.65, smoothForward);
        //float zoom = 1.0;
        //vec3  ro = ta + zoom*normalize(vec3(cos(roty), cos(rotx), sin(roty))); 
        ///////////////////////****************************//////////////////////
        
        float an_x = 10.*iMouse.x/iResolution.x;
        float an_y = 10.*iMouse.y/iResolution.y;
    vec3  ta = vec3( 0.0, 0.65, smoothForward);
        vec3  ro = ta + vec3( 1.5*cos(an_x), 1.5*cos(an_y), 1.5*sin(an_x) ); 
        
        // camera bounce
        float t4 = abs(fract(time*0.5)-0.5)/0.5;
        float bou = -1.0 + 2.0*t4;
        ro += 0.03 //weight
                    *sin(time*12.0+vec3(0.0,2.0,4.0))*smoothstep( 0.85, 1.0, abs(bou) );


        vec3 cw = normalize(ta-ro);
        vec3 cp = vec3(0.0, 1.0,0.0); 
        vec3 cu = normalize( cross(cw,cp) ); 
        vec3 cv =          ( cross(cu,cw) ); 


        vec3 rd = normalize( p.x*cu + p.y*cv + 1.8*cw );

        vec3 col = render( ro, rd, time, uv );

        col = pow( col, vec3(0.4545) );

        tot += col;
    #if AA>1
        }
        tot /= float(AA*AA);
#endif

    fragColor = vec4( tot, 1.0 );
}
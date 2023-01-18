// parmas
struct Material
{
    vec3 	diffuseAlbedo;
    vec3 	specularAlbedo;
    float 	specularPower;
};
Material stem_mat = Material(vec3(0.8196, 0.3922, 0.2078), vec3(0.3), 8.0);

// cloud params
const int _VolumeSteps = 64;
const float _StepSize = 0.05; 
const float _Density = 0.1;
const float _OpacityThreshold = 0.95;

const float _SphereRadius = 1.2;
const float _NoiseFreq = 0.5;
const float _NoiseAmp = 2.0;

const vec4 innerColor = vec4(0.7, 0.7, 0.7, _Density);
const vec4 outerColor = vec4(1.0, 1.0, 1.0, 0.0);

//utils
// Smooth Min
float smin( float a, float b, float k )
{
    float h = max(k-abs(a-b),0.0);
    return min(a, b) - h*h*0.25/k;
}

vec2 smin( vec2 a, vec2 b, float k )
{
    float h = clamp( 0.5+0.5*(b.x-a.x)/k, 0.0, 1.0 );
    return mix( b, a, h ) - k*h*(1.0-h);
}

//Smooth Max ; 实现缺口
float smax( float a, float b, float k )
{
    float h = max(k-abs(a-b),0.0);
    return max(a, b) + h*h*0.25/k;
}

vec2 opU( vec2 d1, vec2 d2 )
{
	return (d1.x<d2.x) ? d1 : d2;
}

mat2 rotMat(float rot)
{
    float cc = cos(rot);
    float ss = sin(rot);
    return mat2(cc, ss, -ss, cc);
}

///////////////////////////////////
//sdf
//球
float sdSphere( vec3 p, float s )
{
    return length(p)-s;
}

//椭圆
float sdEllipsoid( in vec3 p, in vec3 r )
{
    float k0 = length(p/r);
    float k1 = length(p/(r*r));
    return k0*(k0-1.0)/k1;
}

//棍
vec2 sdStick(vec3 p, vec3 a, vec3 b, float r1, float r2)
{
    vec3 pa = p-a, ba = b-a;
	float h = clamp( dot(pa,ba)/dot(ba,ba), 0.0, 1.0 );
	return vec2( length( pa - ba*h ) - mix(r1,r2,h*h*(3.0-2.0*h)), h );
}

vec2 sdSegment( in vec3 p, vec3 a, vec3 b )
{
	vec3 pa = p - a, ba = b - a;
	float h = clamp( dot(pa,ba)/dot(ba,ba), 0.0, 1.0 );
	return vec2( length( pa - ba*h ), h );
}

////////////////////////////////
// noise
vec3 mod289(vec3 x) {
  return x - floor(x * (1.0 / 289.0)) * 289.0;
}

vec4 mod289(vec4 x) {
  return x - floor(x * (1.0 / 289.0)) * 289.0;
}

vec4 permute(vec4 x) {
     return mod289(((x*34.0)+10.0)*x);
     //return mod289(((x*34.0)+1.0)*x);
}
// 2D
vec2 mod289(vec2 x) { return x - floor(x * (1.0 / 289.0)) * 289.0; }
vec3 permute(vec3 x) { return mod289(((x*34.0)+1.0)*x); }

vec4 taylorInvSqrt(vec4 r)
{
  return 1.79284291400159 - 0.85373472095314 * r;
}

// float snoise(vec3 v)
// {
//   const vec2  C = vec2(1.0/6.0,1.0/3.0);
//   const vec4  D = vec4(0.211324865405187,
//                         // (3.0-sqrt(3.0))/6.0
//                         0.366025403784439,
//                         // 0.5*(sqrt(3.0)-1.0)
//                         -0.577350269189626,
//                         // -1.0 + 2.0 * C.x
//                         0.024390243902439);
//                         // 1.0 / 41.0
// //const vec4  D = vec4(0.0, 0.5, 1.0, 2.0);

// // First corner
//   vec3 i  = floor(v + dot(v, C.yyy) );
//   vec3 x0 =   v - i + dot(i, C.xxx) ;

// // Other corners
//   vec3 g = step(x0.yzx, x0.xyz);
//   vec3 l = 1.0 - g;
//   vec3 i1 = min( g.xyz, l.zxy );
//   vec3 i2 = max( g.xyz, l.zxy );

//   //   x0 = x0 - 0.0 + 0.0 * C.xxx;
//   //   x1 = x0 - i1  + 1.0 * C.xxx;
//   //   x2 = x0 - i2  + 2.0 * C.xxx;
//   //   x3 = x0 - 1.0 + 3.0 * C.xxx;
//   vec3 x1 = x0 - i1 + C.xxx;
//   vec3 x2 = x0 - i2 + C.yyy; // 2.0*C.x = 1/3 = C.y
//   vec3 x3 = x0 - D.yyy;      // -1.0+3.0*C.x = -0.5 = -D.y

// // Permutations
//   i = mod289(i); 
//   vec4 p = permute( permute( permute( 
//              i.z + vec4(0.0, i1.z, i2.z, 1.0 ))
//            + i.y + vec4(0.0, i1.y, i2.y, 1.0 )) 
//            + i.x + vec4(0.0, i1.x, i2.x, 1.0 ));

// // Gradients: 7x7 points over a square, mapped onto an octahedron.
// // The ring size 17*17 = 289 is close to a multiple of 49 (49*6 = 294)
//   float n_ = 0.142857142857; // 1.0/7.0
//   vec3  ns = n_ * D.wyz - D.xzx;

//   vec4 j = p - 49.0 * floor(p * ns.z * ns.z);  //  mod(p,7*7)

//   vec4 x_ = floor(j * ns.z);
//   vec4 y_ = floor(j - 7.0 * x_ );    // mod(j,N)

//   vec4 x = x_ *ns.x + ns.yyyy;
//   vec4 y = y_ *ns.x + ns.yyyy;
//   vec4 h = 1.0 - abs(x) - abs(y);

//   vec4 b0 = vec4( x.xy, y.xy );
//   vec4 b1 = vec4( x.zw, y.zw );

//   //vec4 s0 = vec4(lessThan(b0,0.0))*2.0 - 1.0;
//   //vec4 s1 = vec4(lessThan(b1,0.0))*2.0 - 1.0;
//   vec4 s0 = floor(b0)*2.0 + 1.0;
//   vec4 s1 = floor(b1)*2.0 + 1.0;
//   vec4 sh = -step(h, vec4(0.0));

//   vec4 a0 = b0.xzyw + s0.xzyw*sh.xxyy ;
//   vec4 a1 = b1.xzyw + s1.xzyw*sh.zzww ;

//   vec3 p0 = vec3(a0.xy,h.x);
//   vec3 p1 = vec3(a0.zw,h.y);
//   vec3 p2 = vec3(a1.xy,h.z);
//   vec3 p3 = vec3(a1.zw,h.w);

// //Normalise gradients
//   vec4 norm = taylorInvSqrt(vec4(dot(p0,p0), dot(p1,p1), dot(p2, p2), dot(p3,p3)));
//   p0 *= norm.x;
//   p1 *= norm.y;
//   p2 *= norm.z;
//   p3 *= norm.w;

// // Mix final noise value
//   vec4 m = max(0.5 - vec4(dot(x0,x0), dot(x1,x1), dot(x2,x2), dot(x3,x3)), 0.0);
//   m = m * m;
//   return 130.0 * dot( m*m, vec4( dot(p0,x0), dot(p1,x1), 
//                                 dot(p2,x2), dot(p3,x3) ) );
// }

float snoise(vec3 v)
  { 
  const vec2  C = vec2(1.0/6.0, 1.0/3.0) ;
  const vec4  D = vec4(0.0, 0.5, 1.0, 2.0);

  // First corner
  vec3 i  = floor(v + dot(v, C.yyy) );
  vec3 x0 =   v - i + dot(i, C.xxx) ;

  // Other corners
  vec3 g = step(x0.yzx, x0.xyz);	  
  vec3 l = 1.0 - g;
  vec3 i1 = min( g.xyz, l.zxy );
  vec3 i2 = max( g.xyz, l.zxy );

  //   x0 = x0 - 0.0 + 0.0 * C.xxx;
  //   x1 = x0 - i1  + 1.0 * C.xxx;
  //   x2 = x0 - i2  + 2.0 * C.xxx;
  //   x3 = x0 - 1.0 + 3.0 * C.xxx;
  vec3 x1 = x0 - i1 + C.xxx;
  vec3 x2 = x0 - i2 + C.yyy; // 2.0*C.x = 1/3 = C.y
  vec3 x3 = x0 - D.yyy;      // -1.0+3.0*C.x = -0.5 = -D.y

  // Permutations
  i = mod289(i); 
  vec4 p = permute( permute( permute( 
             i.z + vec4(0.0, i1.z, i2.z, 1.0 ))
           + i.y + vec4(0.0, i1.y, i2.y, 1.0 )) 
           + i.x + vec4(0.0, i1.x, i2.x, 1.0 ));

  // Gradients: 7x7 points over a square, mapped onto an octahedron.
  // The ring size 17*17 = 289 is close to a multiple of 49 (49*6 = 294)
  float n_ = 0.142857142857; // 1.0/7.0
  vec3  ns = n_ * D.wyz - D.xzx;

  vec4 j = p - 49.0 * floor(p * ns.z * ns.z);  //  mod(p,7*7)

  vec4 x_ = floor(j * ns.z);
  vec4 y_ = floor(j - 7.0 * x_ );    // mod(j,N)

  vec4 x = x_ *ns.x + ns.yyyy;
  vec4 y = y_ *ns.x + ns.yyyy;
  vec4 h = 1.0 - abs(x) - abs(y);

  vec4 b0 = vec4( x.xy, y.xy );
  vec4 b1 = vec4( x.zw, y.zw );

  //vec4 s0 = vec4(lessThan(b0,0.0))*2.0 - 1.0;
  //vec4 s1 = vec4(lessThan(b1,0.0))*2.0 - 1.0;
  vec4 s0 = floor(b0)*2.0 + 1.0;
  vec4 s1 = floor(b1)*2.0 + 1.0;
  vec4 sh = -step(h, vec4(0.0));

  vec4 a0 = b0.xzyw + s0.xzyw*sh.xxyy ;
  vec4 a1 = b1.xzyw + s1.xzyw*sh.zzww ;

  vec3 p0 = vec3(a0.xy,h.x);
  vec3 p1 = vec3(a0.zw,h.y);
  vec3 p2 = vec3(a1.xy,h.z);
  vec3 p3 = vec3(a1.zw,h.w);

  //Normalise gradients
  vec4 norm = taylorInvSqrt(vec4(dot(p0,p0), dot(p1,p1), dot(p2, p2), dot(p3,p3)));
  p0 *= norm.x;
  p1 *= norm.y;
  p2 *= norm.z;
  p3 *= norm.w;

  // Mix final noise value
  vec4 m = max(0.6 - vec4(dot(x0,x0), dot(x1,x1), dot(x2,x2), dot(x3,x3)), 0.0);
  m = m * m;
  return 42.0 * dot( m*m, vec4( dot(p0,x0), dot(p1,x1), 
                                dot(p2,x2), dot(p3,x3) ) );
}

//----

float snoise(vec2 v) {
    const vec4 C = vec4(0.211324865405187,  // (3.0-sqrt(3.0))/6.0
                        0.366025403784439,  // 0.5*(sqrt(3.0)-1.0)
                        -0.577350269189626,  // -1.0 + 2.0 * C.x
                        0.024390243902439); // 1.0 / 41.0
    vec2 i  = floor(v + dot(v, C.yy) );
    vec2 x0 = v -   i + dot(i, C.xx);
    vec2 i1;
    i1 = (x0.x > x0.y) ? vec2(1.0, 0.0) : vec2(0.0, 1.0);
    vec4 x12 = x0.xyxy + C.xxzz;
    x12.xy -= i1;
    i = mod289(i); // Avoid truncation effects in permutation
    vec3 p = permute( permute( i.y + vec3(0.0, i1.y, 1.0 ))
        + i.x + vec3(0.0, i1.x, 1.0 ));

    vec3 m = max(0.5 - vec3(dot(x0,x0), dot(x12.xy,x12.xy), dot(x12.zw,x12.zw)), 0.0);
    m = m*m ;
    m = m*m ;
    vec3 x = 2.0 * fract(p * C.www) - 1.0;
    vec3 h = abs(x) - 0.5;
    vec3 ox = floor(x + 0.5);
    vec3 a0 = x - ox;
    m *= 1.79284291400159 - 0.85373472095314 * ( a0*a0 + h*h );
    vec3 g;
    g.x  = a0.x  * x0.x  + h.x  * x0.y;
    g.yz = a0.yz * x12.xz + h.yz * x12.yw;
    return 130.0 * dot(m, g);
}

float random (in vec2 _st) {
    return fract(sin(dot(_st.xy,vec2(12.9898,78.633)))*43758.5453123);
}

float noise (in vec2 _st) {
    vec2 i = floor(_st);
    vec2 f = fract(_st);
    float a = random(i);
    float b = random(i + vec2(1.0, 0.0));
    float c = random(i + vec2(0.0, 1.0));
    float d = random(i + vec2(1.0, 1.0));

    vec2 u = f * f * f * f *(3.0 - 2.0 * f);  //+2.0*sin(u_time/10.0)

    return mix(a, b, u.x) +
            (c - a)* u.y * (1.0 - u.x) +
            (d - b) * u.x * u.y;
}

float fbm ( in vec2 _st) {
    float v = 0.0;
    float a = 0.5;
    vec2 shift = vec2(100.0);
    mat2 rot = mat2(cos(0.5), sin(0.5),
                   -sin(0.5), cos(0.5));
  
    for (int i = 0; i < 3; ++i) {
        v += a * noise(_st);
        _st = rot * _st * (4.5) + shift;
        a *= 0.45;
    }
    return v;
}

float fbm(vec3 p)
{
    float f;
    f = 0.5000*snoise( p ); p = p*2.02;
    f += 0.2500*snoise( p ); p = p*2.03;
    f += 0.1250*snoise( p ); p = p*2.01;
    f += 0.0625*snoise( p );
    return f;
}

////////////////////////////////
// add elements
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

vec2 sdMushroom(vec3 p)
{
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
    d1 -= 0.04;
    d1 -= 0.1*exp(-16.0*h);
    vec2 res = vec2(d1, 8.0);

    // head
    float d2 = head( p + vec3(0.06,-1.5,0.0));
    vec2 res2 = vec2(d2, 7.0);

    // mix head and stem
    //d1 = smin( d1, d3, 0.2 );
    //d1 *= 0.75; 
    res = opU(res2, res);
        
    return res;
}

vec2 sdTerrian(in vec3 pos, float atime)
{
    float floorHeight = -0.1 //基础高度
                                     - 0.05*(sin(pos.x*2.0)+sin(pos.z*2.0)); //地形
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

////////////////////////////////
//绘制物体
vec2 map( in vec3 pos, float atime )
{
    
    vec2 res = vec2(-1.,-1.);
    
    //yMove
    float t = fract(atime); //[0,1]的上下波动函数
    float bounceWave = 4.0*t*(1.0-t);  //对t进行了曲线化，为了制作球体的上下跳动
    float yMove = pow(bounceWave,2.0-bounceWave) + 0.1; //下快上慢
    
    //zMove
    float zMove = floor(atime) + pow(t,0.7) -1.0 ; //非线性的smooth移动
        
     //xMove
    float tt = abs(fract(atime*0.5)-0.5)/0.5;
    float xMove = 0.5*(-1.0 + 2.0*tt);
     
    vec3 cen = vec3( xMove, yMove, zMove);//导入函数，画出球心的跳动感
    
    //Coordinate
    vec3 basic = pos-cen; 
    basic.xy = rotMat(-xMove*0.3) * basic.xy;
    
    vec3 symmBody =  vec3(  abs(basic.x), basic.yz);
 
    
    //body: 尝试了形变效果，Q弹效果
    float sy = 0.9 + 0.1*bounceWave;
    float compress = 1.0-smoothstep(0.0,0.4,bounceWave);
    sy = sy*(1.0-compress) + compress;
    float sx = 1./sy;
    
    float dBody = sdEllipsoid( basic , vec3(0.2*sx, 0.2*sy, 0.2) );

    //arm
    vec3 armPos = vec3(0.2, 0., 0.);
    float dArm = sdSphere( symmBody - armPos, 0.07);
    
    dBody = smin(dBody, dArm, 0.05);
    
    //mouth
    {
    float smell = 23.* basic.x*basic.x;
    vec3 mouthPos = vec3(0., -0.03 +smell, 0.17);
    float dMouth = sdEllipsoid( basic - mouthPos, vec3(0.1, 0.11, 0.15)*0.3 );
    
    dBody = smax(dBody, -dMouth, 0.01);
    vec2 bodyObj = vec2(dBody, 2.0);
    res = bodyObj;
    }
    
    // leg
    //删去了随着上下形变的效果。TODO：前后摇摆时脚的位置应该发生变化
    {
    float sy = 0.5 + 0.3*bounceWave;
    float compress = 1.0-smoothstep(0.0,0.4,bounceWave);
    sy = sy*(1.0-compress) + compress;
    float sx = 1./sy;
    
    vec3 legPos = vec3(0.11, -0.17, 0.05);
    legPos = symmBody - legPos;
    legPos.y += 0.2* legPos.x*legPos.x; //脚底有比较扁
    
    float t6 = cos(6.2831*(atime*0.5+0.25));
    legPos.xz =rotMat(-1.0) * legPos.xz ;    //八字型
    legPos.xy =rotMat(t6*sign(basic.x)) * legPos.xy ; //随时间前后摇摆
        
    float dleg = sdEllipsoid( legPos, vec3(0.43, 0.18, 0.28)*0.3 );
    vec2 legObj = vec2(dleg, 3.0);
    
    res = opU(res, legObj);
    }
    
    
    //eye
    vec3 eyePos = vec3(0.06, 0.05, 0.17);
    eyePos = symmBody - eyePos;
    
    eyePos.yz = rotMat(0.2)*eyePos.yz;//更贴合脸部
    float dEye = sdEllipsoid( eyePos, vec3(0.15, 0.3, 0.1)*0.2 );
    vec2 eyeObj = vec2(dEye, 4.0);
    res = opU(res, eyeObj);
    
     //eyeball
    vec3 eyeballPos = vec3(0.065, 0.07, 0.17);
    eyeballPos = symmBody - eyeballPos;
    
    float dEyeball = sdSphere( eyeballPos, 0.02 );
    vec2 eyeballObj = vec2(dEyeball, 5.0);
    res = opU(res, eyeballObj);
    
    
    // ground
    vec2 floorData = sdTerrian(pos, atime);
    float floorHeight = floorData.x;
    float dFloor = floorData.y;
    
    // bubbles
    
    vec3 bubbleArea = vec3( mod(abs(pos.x),3.0),pos.y,mod(pos.z+1.5,3.0)-1.5);
        //随机生成
    vec2 id = vec2( floor(pos.x/3.0), floor((pos.z+1.5)/3.0) );
    float fid = id.x*11.1 + id.y*31.7;
        //飘起来的效果
    float fy = fract(fid*1.312+atime*0.1);
    float y = -1.0+4.0*fy;
    
    vec3  rad = vec3(0.7,1.0+0.5*sin(fid),0.7);
    rad -= 0.1*(sin(pos.x*3.0)+sin(pos.y*4.0)+sin(pos.z*5.0));    
    
        //smoothly  change the size when fly
   float siz = 4.0*fy*(1.0-fy);
    float dTree = sdEllipsoid( bubbleArea-vec3(2.0,y,0.0), rad*siz );
        //添加突出效果
    float bubbleTexture = 0.2*(-1.0+2.0*smoothstep(-0.2,0.2, sin(18.0*pos.x)+sin(18.0*pos.y)+sin(18.0*pos.z))); 
    dTree += 0.01 * bubbleTexture;
    
    dTree *= 0.6;
    dTree = min(dTree,2.0);

    dFloor = smin( dFloor, dTree, 0.32 );
    vec2 floorObj = vec2(dFloor, 1.0);
    
    res = opU(res, floorObj);
    
    
    //candy
    {
    float fs = 5.0;
    vec3 qos = fs*vec3(pos.x, pos.y-floorHeight, pos.z );
    vec2 id = vec2( floor(qos.x+0.5), floor(qos.z+0.5) );
    
    vec3 vp = vec3( fract(qos.x+0.5)-0.5,qos.y,fract(qos.z+0.5)-0.5);
    vp.xz += 0.1*cos( id.x*130.143 + id.y*120.372 + vec2(0.0,2.0) );
    
    float den = sin(id.x*0.1+sin(id.y*0.091))+sin(id.y*0.1);
    float fid = id.x*0.143 + id.y*0.372;
    float ra = smoothstep(0.0,0.1,den*0.1+fract(fid)-0.95);
    
    float dCandy = sdSphere( vp, 0.35*ra )/fs;
    vec2 candyObj = vec2(dCandy, 6.0);
    
    res = opU(res, candyObj);   

    // mushroom
    vec3 q = pos;
    q.xz = mod(q.xz - vec2(2.5), 5.0) - vec2(2.5); // repeat
    q.y -= 1.05;
    vec2 mushroomObj = sdMushroom(q);
    res = opU(res, mushroomObj);
    }

    return res;
}

//光线跟踪技术
vec2 castRay( in vec3 ro, in vec3 rd, float time )
{
    vec2 res = vec2(-1.0,-1.0);

    float tmin = 0.5;
    float tmax = 20.0;
    
    float t = tmin;
    for( int i=0; i<256 && t<tmax; i++ )
    {
        vec2 h = map( ro+rd*t, time );
        if( abs(h.x)<(0.0005*t) )
        { 
            res = vec2(t,h.y); 
            break;
        }
        t += h.x;
    }
    
    return res;
}


//归一化函数
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

//颜色渲染
vec3 render( in vec3 ro, in vec3 rd, float time )
{ 
    // sky dome
    vec3 col = vec3(0.5, 0.8, 0.9) - max(rd.y,0.0)*0.5;
    vec3 hitPos;
    col += castRayVolume(ro, rd*0.7, vec4(0.0), hitPos).rgb; // rd*stepSize

    
    vec2 res = castRay(ro,rd, time);
    if( res.y>-0.5 )
    {
        float t = res.x;
        vec3 pos = ro + t*rd;
        vec3 nor = calcNormal( pos, time );
        vec3 ref = reflect( rd, nor );
        
        //渲染颜色
		col = vec3(0.2);
        float ks = 1.0;

        if (res.y > 7.5) // mushroom stem
        {
            col = vec3(0.2588, 0.1098, 0.0392);
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
        else if (res.y > 5.5)  //candy
        {
            col = vec3(0.14,0.048,0.0); 
             vec2 id = floor(5.0*pos.xz+0.5);
		     col += 0.036*cos((id.x*11.1+id.y*37.341) + vec3(0.0,1.0,2.0) );
             col = max(col,0.0);
        }
        else if (res.y > 4.5)
        {
            col = vec3(0.14,0.048,0.0); 
        }
        else if( res.y>3.5 ) // eye
        {  // todo: 渐变色:需要返回相对坐标不能使用绝对坐标
            vec3 black = vec3(0.0);
            vec3 blue = vec3(0.0,0.0,1.0);
            col = (1.0 - vec3(smoothstep(0.9,1.04,pos.y))) * blue;
            vec2 circleP = vec2(0.07, 1.06);
            float xx = (abs(pos.x) - circleP.x)*(abs(pos.x) - circleP.x);
            float yy = (pos.y - circleP.y)*(pos.y - circleP.y);
            //if ( xx+yy < 0.0003){col = vec3(1.0);}
            
            col = vec3(0.0);
        } 
        else if( res.y>2.5 ) // leg
        { 
            col = vec3(0.4,0.0,0.02);
        } 
        else if( res.y>1.5 ) // body
        { 
            col = vec3(0.5,0.07,0.1);
        }
		else // terrain
        {
            col = vec3(0.3961, 0.2118, 0.2196);
            //col += 0.5*snoise(vec2(fract(pos.x), fract(pos.y)));
            
            //格子颜色
            float wave  = sin(18.0*pos.x)+sin(18.0*pos.y)+sin(18.0*pos.z); //生成黑白相间的颜色
            float f = 0.2*(-1.0+2.0*smoothstep(-0.2,0.2,wave)); //让wave变得更加锋利，颜色变化更剧烈
            col += f*vec3(0.06,0.06,0.02);
            
            //提高亮度：对阳光反射亮度进行了加强
            ks = 0.5 + pos.y*0.15;
        }
        
        // lighting
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
        
        col = col*lin;
		col += sun_spe*vec3(8.10,6.00,4.20)*sun_sha;
        
        if (res.y > 7.5)
        {
            vec3 lightColor = vec3(1.0);
            
            float t = clamp(0.5, 0.2, 1.0);
            vec3 light = t * lightColor * calcTransmittance(pos+nor*vec3(0.01), sun_lig, 0.01, 10.0, 2.0, time);
            light += (1.0 - t) * calcTransmittance(pos+nor*vec3(0.01), rd, 0.01, 10.0, 0.5, time);
            col =  light * stem_mat.diffuseAlbedo;
            col += light * stem_mat.specularAlbedo * pow(max(0.0, dot(reflect(sun_lig,nor),rd)), 4.0);
        }

        col = mix( col, vec3(0.5,0.7,0.9), 1.0-exp( -0.0001*t*t*t ) );
    }

    return col;
}

void mainImage( out vec4 fragColor, in vec2 fragCoord )
{
    //转换坐标，实现1. 位于中心 2.比例与画布长宽相同
    vec2 p = (-iResolution.xy + 2.0*fragCoord)/iResolution.y;
    float time = iTime;
    //float time = 1.;
    time *= 0.9;

    // camera	
    //float an = 0.;
    //float an = 10.*iMouse.x/iResolution.x;
    float rotx = 2.5 + (iMouse.y / iResolution.y)*4.0;
    float roty = 2.5 + (iMouse.x / iResolution.x)*4.0;
    
    float forwardWave = sin(0.5*time);
    float smoothForward = -1.
                                            +time*1.0 - 0.4*forwardWave; //虽然依旧是线性前进但是有个动态的微小波动更显真实
    vec3  ta = vec3( 0.0, 0.65, smoothForward);
    //vec3  ta = vec3( 0.0, 0.95, 0.);  //如果用这个就不会发生拖动的时候的位置移动
    float zoom = 1.0;
    vec3  ro = ta + zoom*normalize(vec3(cos(roty), cos(rotx), sin(roty))); // 摄像机位置
    
    // camera bounce
    float t4 = abs(fract(time*0.5)-0.5)/0.5;
    float bou = -1.0 + 2.0*t4;
    ro += 0.03 //weight
                *sin(time*12.0+vec3(0.0,2.0,4.0))*smoothstep( 0.85, 1.0, abs(bou) );

    // 相机方向的变换矩阵
	vec3 cw = normalize(ta-ro); //(lookatPoint - original) 相机方向向量：前向向量
	vec3 cp = vec3(0.0, 1.0,0.0); 
	vec3 cu = normalize( cross(cw,cp) ); // 利用cp求出向右的向量
	vec3 cv =          ( cross(cu,cw) ); //反推回真正的向上的向量

    //得到最终的相机方向
    vec3 rd = normalize( p.x*cu + p.y*cv + 1.8*cw ); //得到可以移动的

    vec3 col = render( ro, rd, time );

    col = pow( col, vec3(0.4545) ); //加强颜色效果

    fragColor = vec4( col, 1.0 );
}
import{s as v,_ as g}from"./nrsr_share_image-bef09b5e.mjs";import{_ as m,o as n,c as r,a as e,t as c,F as y,r as $,d as a,f as k,p as x,g as b,h as P,i as I,j as S,u as j,b as p}from"./entry-233ab20d.mjs";var w=[{name:"Pellegrini Peter",perc:60.5,perc_floor:60,perc_tens:5,family_name:"Pellegrini",other_names:"Peter"},{name:"Kor\u010Dok Ivan",perc:39.5,perc_floor:39,perc_tens:5,family_name:"Kor\u010Dok",other_names:"Ivan"}],D=[{date:"2024-03-02"}];const V={data:function(){return{candidates:w,date:new Date(D[0].date).toLocaleDateString("sk")}}},B=a(" \u0160ance na v\xFDhru ve volb\xE1ch "),C=a("dle s\xE1zkov\xFDch kancel\xE1\u0159\xED "),E={class:"fs-5 fw-bolder"},F={class:"container"},L={class:"row"},N={class:"card text-center mb-2"},z={class:"card-header card-title"},O={class:"card-body"},A={class:"number fs-2"},K={key:0},M=a(" %");function H(t,_,s,i,d,u){return n(),r("div",null,[e("h3",null,[B,e("small",null,[C,e("span",E,c(t.date),1)])]),e("div",F,[e("div",L,[(n(!0),r(y,null,$(t.candidates,(o,l)=>(n(),r("div",{key:l,class:"col-xl-3 col-lg-4 col-sm-6"},[e("div",N,[e("h4",z,[a(c(o.family_name)+" ",1),e("small",null,c(o.other_names),1)]),e("div",O,[e("div",A,[a(c(o.perc_floor),1),o.perc_floor<5?(n(),r("small",K,"."+c(o.perc_tens),1)):k("",!0),M])])])]))),128))])])])}var R=m(V,[["render",H]]);const T={},h=t=>(x("data-v-536f1bf9"),t=t(),b(),t),U={id:"image-wrapper"},q=h(()=>e("h3",null,[a(" V\xFDvoj \u0161ance na prezidenta "),e("small",null,"pod\u013Ea st\xE1vkov\xFDch kancel\xE1ri\xED ")],-1)),G=h(()=>e("img",{src:P,class:"image"},null,-1)),J=h(()=>e("img",{src:I,class:"image-small"},null,-1)),Q=[q,G,J];function W(t,_){return n(),r("div",U,Q)}var X=m(T,[["render",W],["__scopeId","data-v-536f1bf9"]]);const Y=S({name:"index",setup(t,{expose:_}){_();const s="https://mandaty.sk/",i="493242628099686";j({title:"Mand\xE1ty.sk",meta:[{hid:"og:name",property:"og:image",content:s+v.filename},{hid:"og:url",property:"og:url",content:s},{hid:"og:type",property:"og:type",content:"website"},{hid:"og:title",property:"og:title",content:"Mand\xE1ty.sk"},{hid:"og:description",property:"og:description",content:"V\xFDvoj volebn\xFDch modelov pod\u013Ea prieskumov verejnej mienky"},{hid:"fb:app_id",property:"fb:app_id",content:i}],link:[{rel:"icon",type:"image/x-icon",href:s+"favicon.svg"}]});const d={BASE_URL:s,FB_APP_ID:i};return Object.defineProperty(d,"__isScriptSetup",{enumerable:!1,value:!0}),d}}),Z={class:"container"},ee=e("h1",{class:"pt-4"},"Prezident/ka 2024",-1),te=e("hr",null,null,-1);function oe(t,_,s,i,d,u){const o=g,l=R,f=X;return n(),r("div",Z,[p(o),ee,p(l),te,p(f)])}var re=m(Y,[["render",oe]]);export{re as default};
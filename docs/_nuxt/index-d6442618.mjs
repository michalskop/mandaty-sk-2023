import{s as v,_ as g}from"./nrsr_share_image-c04ff817.mjs";import{_ as m,o as s,c as r,a as e,t as c,F as y,r as $,d as a,f as k,p as x,g as b,h as P,i as I,j as S,u as j,b as p}from"./entry-96fdcc3b.mjs";var w=[{name:"Ne",perc:48.2,perc_floor:48,perc_tens:2,family_name:"Ne",other_names:""},{name:"Pellegrini Peter",perc:32.6,perc_floor:32,perc_tens:6,family_name:"Pellegrini",other_names:"Peter"},{name:"Kor\u010Dok Ivan",perc:19.3,perc_floor:19,perc_tens:3,family_name:"Kor\u010Dok",other_names:"Ivan"}],D=[{date:"2024-03-01"}];const N={data:function(){return{candidates:w,date:new Date(D[0].date).toLocaleDateString("sk")}}},V=a(" \u0160ance na v\xFDhru ve volb\xE1ch "),B=a("dle s\xE1zkov\xFDch kancel\xE1\u0159\xED "),C={class:"fs-5 fw-bolder"},E={class:"container"},F={class:"row"},L={class:"card text-center mb-2"},z={class:"card-header card-title"},O={class:"card-body"},A={class:"number fs-2"},K={key:0},M=a(" %");function H(t,_,n,i,d,u){return s(),r("div",null,[e("h3",null,[V,e("small",null,[B,e("span",C,c(t.date),1)])]),e("div",E,[e("div",F,[(s(!0),r(y,null,$(t.candidates,(o,l)=>(s(),r("div",{key:l,class:"col-xl-3 col-lg-4 col-sm-6"},[e("div",L,[e("h4",z,[a(c(o.family_name)+" ",1),e("small",null,c(o.other_names),1)]),e("div",O,[e("div",A,[a(c(o.perc_floor),1),o.perc_floor<5?(s(),r("small",K,"."+c(o.perc_tens),1)):k("",!0),M])])])]))),128))])])])}var R=m(N,[["render",H]]);const T={},h=t=>(x("data-v-536f1bf9"),t=t(),b(),t),U={id:"image-wrapper"},q=h(()=>e("h3",null,[a(" V\xFDvoj \u0161ance na prezidenta "),e("small",null,"pod\u013Ea st\xE1vkov\xFDch kancel\xE1ri\xED ")],-1)),G=h(()=>e("img",{src:P,class:"image"},null,-1)),J=h(()=>e("img",{src:I,class:"image-small"},null,-1)),Q=[q,G,J];function W(t,_){return s(),r("div",U,Q)}var X=m(T,[["render",W],["__scopeId","data-v-536f1bf9"]]);const Y=S({name:"index",setup(t,{expose:_}){_();const n="https://mandaty.sk/",i="493242628099686";j({title:"Mand\xE1ty.sk",meta:[{hid:"og:name",property:"og:image",content:n+v.filename},{hid:"og:url",property:"og:url",content:n},{hid:"og:type",property:"og:type",content:"website"},{hid:"og:title",property:"og:title",content:"Mand\xE1ty.sk"},{hid:"og:description",property:"og:description",content:"V\xFDvoj volebn\xFDch modelov pod\u013Ea prieskumov verejnej mienky"},{hid:"fb:app_id",property:"fb:app_id",content:i}],link:[{rel:"icon",type:"image/x-icon",href:n+"favicon.svg"}]});const d={BASE_URL:n,FB_APP_ID:i};return Object.defineProperty(d,"__isScriptSetup",{enumerable:!1,value:!0}),d}}),Z={class:"container"},ee=e("h1",{class:"pt-4"},"Prezident/ka 2024",-1),te=e("hr",null,null,-1);function oe(t,_,n,i,d,u){const o=g,l=R,f=X;return s(),r("div",Z,[p(o),ee,p(l),te,p(f)])}var re=m(Y,[["render",oe]]);export{re as default};

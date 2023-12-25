
<stdin>:	file format elf64-amdgpu

Disassembly of section .text:

0000000000001600 <r_20_20>:
	s_load_b128 s[0:3], s[0:1], null                           // 000000001600: F4080000 F8000000
	s_mov_b32 s6, s15                                          // 000000001608: BE86000F
	s_ashr_i32 s7, s15, 31                                     // 00000000160C: 86079F0F
	s_add_i32 s8, s15, -1                                      // 000000001610: 8108C10F
	s_lshl_b64 s[4:5], s[6:7], 2                               // 000000001614: 84848206
	s_mov_b32 s7, 0                                            // 000000001618: BE870080
	s_waitcnt lgkmcnt(0)                                       // 00000000161C: BF89FC07
	s_add_u32 s2, s2, s4                                       // 000000001620: 80020402
	s_addc_u32 s3, s3, s5                                      // 000000001624: 82030503
	s_add_u32 s2, s2, 0xffffffb4                               // 000000001628: 8002FF02 FFFFFFB4
	s_addc_u32 s3, s3, -1                                      // 000000001630: 8203C103
	s_cmp_lt_i32 s8, 18                                        // 000000001634: BF049208
	s_mov_b32 s8, 0                                            // 000000001638: BE880080
	s_cbranch_scc1 2                                           // 00000000163C: BFA20002 <r_20_20+0x48>
	s_load_b32 s8, s[2:3], null                                // 000000001640: F4000201 F8000000
	s_cmp_lt_i32 s6, 18                                        // 000000001648: BF049206
	s_cbranch_scc1 2                                           // 00000000164C: BFA20002 <r_20_20+0x58>
	s_load_b32 s7, s[2:3], 0x4                                 // 000000001650: F40001C1 F8000004
	s_add_i32 s10, s6, 1                                       // 000000001658: 810A8106
	s_mov_b32 s9, 0                                            // 00000000165C: BE890080
	s_cmp_lt_i32 s10, 18                                       // 000000001660: BF04920A
	s_mov_b32 s10, 0                                           // 000000001664: BE8A0080
	s_cbranch_scc1 2                                           // 000000001668: BFA20002 <r_20_20+0x74>
	s_load_b32 s10, s[2:3], 0x8                                // 00000000166C: F4000281 F8000008
	s_add_i32 s11, s6, 2                                       // 000000001674: 810B8206
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000001678: BF870009
	s_cmp_lt_i32 s11, 18                                       // 00000000167C: BF04920B
	s_cbranch_scc1 2                                           // 000000001680: BFA20002 <r_20_20+0x8c>
	s_load_b32 s9, s[2:3], 0xc                                 // 000000001684: F4000241 F800000C
	s_add_i32 s12, s6, 3                                       // 00000000168C: 810C8306
	s_mov_b32 s11, 0                                           // 000000001690: BE8B0080
	s_cmp_lt_i32 s12, 18                                       // 000000001694: BF04920C
	s_mov_b32 s12, 0                                           // 000000001698: BE8C0080
	s_cbranch_scc1 2                                           // 00000000169C: BFA20002 <r_20_20+0xa8>
	s_load_b32 s12, s[2:3], 0x10                               // 0000000016A0: F4000301 F8000010
	s_add_i32 s13, s6, 4                                       // 0000000016A8: 810D8406
	s_delay_alu instid0(SALU_CYCLE_1)                          // 0000000016AC: BF870009
	s_cmp_lt_i32 s13, 18                                       // 0000000016B0: BF04920D
	s_cbranch_scc1 2                                           // 0000000016B4: BFA20002 <r_20_20+0xc0>
	s_load_b32 s11, s[2:3], 0x14                               // 0000000016B8: F40002C1 F8000014
	s_add_i32 s14, s6, 5                                       // 0000000016C0: 810E8506
	s_mov_b32 s13, 0                                           // 0000000016C4: BE8D0080
	s_cmp_lt_i32 s14, 18                                       // 0000000016C8: BF04920E
	s_mov_b32 s14, 0                                           // 0000000016CC: BE8E0080
	s_cbranch_scc1 2                                           // 0000000016D0: BFA20002 <r_20_20+0xdc>
	s_load_b32 s14, s[2:3], 0x18                               // 0000000016D4: F4000381 F8000018
	s_add_i32 s15, s6, 6                                       // 0000000016DC: 810F8606
	s_delay_alu instid0(SALU_CYCLE_1)                          // 0000000016E0: BF870009
	s_cmp_lt_i32 s15, 18                                       // 0000000016E4: BF04920F
	s_cbranch_scc1 2                                           // 0000000016E8: BFA20002 <r_20_20+0xf4>
	s_load_b32 s13, s[2:3], 0x1c                               // 0000000016EC: F4000341 F800001C
	s_add_i32 s16, s6, 7                                       // 0000000016F4: 81108706
	s_mov_b32 s15, 0                                           // 0000000016F8: BE8F0080
	s_cmp_lt_i32 s16, 18                                       // 0000000016FC: BF049210
	s_mov_b32 s16, 0                                           // 000000001700: BE900080
	s_cbranch_scc1 2                                           // 000000001704: BFA20002 <r_20_20+0x110>
	s_load_b32 s16, s[2:3], 0x20                               // 000000001708: F4000401 F8000020
	s_add_i32 s17, s6, 8                                       // 000000001710: 81118806
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000001714: BF870009
	s_cmp_lt_i32 s17, 18                                       // 000000001718: BF049211
	s_cbranch_scc1 2                                           // 00000000171C: BFA20002 <r_20_20+0x128>
	s_load_b32 s15, s[2:3], 0x24                               // 000000001720: F40003C1 F8000024
	s_add_i32 s18, s6, 9                                       // 000000001728: 81128906
	s_mov_b32 s17, 0                                           // 00000000172C: BE910080
	s_cmp_lt_i32 s18, 18                                       // 000000001730: BF049212
	s_mov_b32 s18, 0                                           // 000000001734: BE920080
	s_cbranch_scc1 2                                           // 000000001738: BFA20002 <r_20_20+0x144>
	s_load_b32 s18, s[2:3], 0x28                               // 00000000173C: F4000481 F8000028
	s_add_i32 s19, s6, 10                                      // 000000001744: 81138A06
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000001748: BF870009
	s_cmp_lt_i32 s19, 18                                       // 00000000174C: BF049213
	s_cbranch_scc1 2                                           // 000000001750: BFA20002 <r_20_20+0x15c>
	s_load_b32 s17, s[2:3], 0x2c                               // 000000001754: F4000441 F800002C
	s_add_i32 s20, s6, 11                                      // 00000000175C: 81148B06
	s_mov_b32 s19, 0                                           // 000000001760: BE930080
	s_cmp_lt_i32 s20, 18                                       // 000000001764: BF049214
	s_mov_b32 s20, 0                                           // 000000001768: BE940080
	s_cbranch_scc1 2                                           // 00000000176C: BFA20002 <r_20_20+0x178>
	s_load_b32 s20, s[2:3], 0x30                               // 000000001770: F4000501 F8000030
	s_add_i32 s21, s6, 12                                      // 000000001778: 81158C06
	s_delay_alu instid0(SALU_CYCLE_1)                          // 00000000177C: BF870009
	s_cmp_lt_i32 s21, 18                                       // 000000001780: BF049215
	s_cbranch_scc1 2                                           // 000000001784: BFA20002 <r_20_20+0x190>
	s_load_b32 s19, s[2:3], 0x34                               // 000000001788: F40004C1 F8000034
	s_add_i32 s22, s6, 13                                      // 000000001790: 81168D06
	s_mov_b32 s21, 0                                           // 000000001794: BE950080
	s_cmp_lt_i32 s22, 18                                       // 000000001798: BF049216
	s_mov_b32 s22, 0                                           // 00000000179C: BE960080
	s_cbranch_scc1 2                                           // 0000000017A0: BFA20002 <r_20_20+0x1ac>
	s_load_b32 s22, s[2:3], 0x38                               // 0000000017A4: F4000581 F8000038
	s_add_i32 s23, s6, 14                                      // 0000000017AC: 81178E06
	s_delay_alu instid0(SALU_CYCLE_1)                          // 0000000017B0: BF870009
	s_cmp_lt_i32 s23, 18                                       // 0000000017B4: BF049217
	s_cbranch_scc1 2                                           // 0000000017B8: BFA20002 <r_20_20+0x1c4>
	s_load_b32 s21, s[2:3], 0x3c                               // 0000000017BC: F4000541 F800003C
	s_add_i32 s24, s6, 15                                      // 0000000017C4: 81188F06
	s_mov_b32 s23, 0                                           // 0000000017C8: BE970080
	s_cmp_lt_i32 s24, 18                                       // 0000000017CC: BF049218
	s_mov_b32 s24, 0                                           // 0000000017D0: BE980080
	s_cbranch_scc1 2                                           // 0000000017D4: BFA20002 <r_20_20+0x1e0>
	s_load_b32 s24, s[2:3], 0x40                               // 0000000017D8: F4000601 F8000040
	s_add_i32 s25, s6, 16                                      // 0000000017E0: 81199006
	s_delay_alu instid0(SALU_CYCLE_1)                          // 0000000017E4: BF870009
	s_cmp_lt_i32 s25, 18                                       // 0000000017E8: BF049219
	s_cbranch_scc1 2                                           // 0000000017EC: BFA20002 <r_20_20+0x1f8>
	s_load_b32 s23, s[2:3], 0x44                               // 0000000017F0: F40005C1 F8000044
	s_add_i32 s26, s6, 17                                      // 0000000017F8: 811A9106
	s_mov_b32 s25, 0                                           // 0000000017FC: BE990080
	s_cmp_lt_i32 s26, 18                                       // 000000001800: BF04921A
	s_mov_b32 s26, 0                                           // 000000001804: BE9A0080
	s_cbranch_scc1 2                                           // 000000001808: BFA20002 <r_20_20+0x214>
	s_load_b32 s26, s[2:3], 0x48                               // 00000000180C: F4000681 F8000048
	s_cmp_gt_u32 s6, 0x7fffffed                                // 000000001814: BF08FF06 7FFFFFED
	s_cbranch_scc1 2                                           // 00000000181C: BFA20002 <r_20_20+0x228>
	s_load_b32 s25, s[2:3], 0x4c                               // 000000001820: F4000641 F800004C
	s_waitcnt lgkmcnt(0)                                       // 000000001828: BF89FC07
	v_add_f32_e64 v0, s8, 0                                    // 00000000182C: D5030000 00010008
	s_add_u32 s0, s0, s4                                       // 000000001834: 80000400
	s_addc_u32 s1, s1, s5                                      // 000000001838: 82010501
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 00000000183C: BF870091
	v_dual_mov_b32 v1, 0 :: v_dual_add_f32 v0, s7, v0          // 000000001840: CA080080 01000007
	v_add_f32_e32 v0, s10, v0                                  // 000000001848: 0600000A
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 00000000184C: BF870091
	v_add_f32_e32 v0, s9, v0                                   // 000000001850: 06000009
	v_add_f32_e32 v0, s12, v0                                  // 000000001854: 0600000C
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001858: BF870091
	v_add_f32_e32 v0, s11, v0                                  // 00000000185C: 0600000B
	v_add_f32_e32 v0, s14, v0                                  // 000000001860: 0600000E
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001864: BF870091
	v_add_f32_e32 v0, s13, v0                                  // 000000001868: 0600000D
	v_add_f32_e32 v0, s16, v0                                  // 00000000186C: 06000010
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001870: BF870091
	v_add_f32_e32 v0, s15, v0                                  // 000000001874: 0600000F
	v_add_f32_e32 v0, s18, v0                                  // 000000001878: 06000012
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 00000000187C: BF870091
	v_add_f32_e32 v0, s17, v0                                  // 000000001880: 06000011
	v_add_f32_e32 v0, s20, v0                                  // 000000001884: 06000014
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001888: BF870091
	v_add_f32_e32 v0, s19, v0                                  // 00000000188C: 06000013
	v_add_f32_e32 v0, s22, v0                                  // 000000001890: 06000016
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001894: BF870091
	v_add_f32_e32 v0, s21, v0                                  // 000000001898: 06000015
	v_add_f32_e32 v0, s24, v0                                  // 00000000189C: 06000018
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 0000000018A0: BF870091
	v_add_f32_e32 v0, s23, v0                                  // 0000000018A4: 06000017
	v_add_f32_e32 v0, s26, v0                                  // 0000000018A8: 0600001A
	s_delay_alu instid0(VALU_DEP_1)                            // 0000000018AC: BF870001
	v_add_f32_e32 v0, s25, v0                                  // 0000000018B0: 06000019
	global_store_b32 v1, v0, s[0:1]                            // 0000000018B4: DC6A0000 00000001
	s_nop 0                                                    // 0000000018BC: BF800000
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)                       // 0000000018C0: BFB60003
	s_endpgm                                                   // 0000000018C4: BFB00000

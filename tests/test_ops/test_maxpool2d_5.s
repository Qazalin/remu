
<stdin>:	file format elf64-amdgpu

Disassembly of section .text:

0000000000001700 <r_1408_5_5_5>:
	s_load_b128 s[0:3], s[0:1], null                           // 000000001700: F4080000 F8000000
	s_mul_i32 s4, s15, 0x8c                                    // 000000001708: 9604FF0F 0000008C
	s_mul_i32 s6, s14, 5                                       // 000000001710: 9606850E
	s_ashr_i32 s5, s4, 31                                      // 000000001714: 86059F04
	v_mov_b32_e32 v1, 0                                        // 000000001718: 7E020280
	s_lshl_b64 s[4:5], s[4:5], 2                               // 00000000171C: 84848204
	s_waitcnt lgkmcnt(0)                                       // 000000001720: BF89FC07
	s_add_u32 s4, s2, s4                                       // 000000001724: 80040402
	s_addc_u32 s5, s3, s5                                      // 000000001728: 82050503
	s_ashr_i32 s7, s6, 31                                      // 00000000172C: 86079F06
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000001730: BF870499
	s_lshl_b64 s[2:3], s[6:7], 2                               // 000000001734: 84828206
	s_add_u32 s2, s4, s2                                       // 000000001738: 80020204
	s_addc_u32 s3, s5, s3                                      // 00000000173C: 82030305
	s_clause 0x4                                               // 000000001740: BF850004
	s_load_b128 s[4:7], s[2:3], null                           // 000000001744: F4080101 F8000000
	s_load_b32 s12, s[2:3], 0x10                               // 00000000174C: F4000301 F8000010
	s_load_b128 s[8:11], s[2:3], 0x70                          // 000000001754: F4080201 F8000070
	s_load_b32 s13, s[2:3], 0x80                               // 00000000175C: F4000341 F8000080
	s_load_b128 s[16:19], s[2:3], 0xe0                         // 000000001764: F4080401 F80000E0
	s_waitcnt lgkmcnt(0)                                       // 00000000176C: BF89FC07
	v_max_f32_e64 v0, s4, s4                                   // 000000001770: D5100000 00000804
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001778: BF870091
	v_max_f32_e32 v0, 0xff800000, v0                           // 00000000177C: 200000FF FF800000
	v_max3_f32 v0, s6, s5, v0                                  // 000000001784: D61C0000 04000A06
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)// 00000000178C: BF8700C1
	v_max3_f32 v0, s12, s7, v0                                 // 000000001790: D61C0000 04000E0C
	s_clause 0x1                                               // 000000001798: BF850001
	s_load_b32 s12, s[2:3], 0xf0                               // 00000000179C: F4000301 F80000F0
	s_load_b128 s[4:7], s[2:3], 0x150                          // 0000000017A4: F4080101 F8000150
	v_max3_f32 v0, s9, s8, v0                                  // 0000000017AC: D61C0000 04001009
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 0000000017B4: BF870091
	v_max3_f32 v0, s11, s10, v0                                // 0000000017B8: D61C0000 0400140B
	v_max3_f32 v0, s16, s13, v0                                // 0000000017C0: D61C0000 04001A10
	s_clause 0x1                                               // 0000000017C8: BF850001
	s_load_b32 s13, s[2:3], 0x160                              // 0000000017CC: F4000341 F8000160
	s_load_b128 s[8:11], s[2:3], 0x1c0                         // 0000000017D4: F4080201 F80001C0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 0000000017DC: BF8700A1
	v_max3_f32 v0, s18, s17, v0                                // 0000000017E0: D61C0000 04002212
	s_waitcnt lgkmcnt(0)                                       // 0000000017E8: BF89FC07
	v_max3_f32 v0, s12, s19, v0                                // 0000000017EC: D61C0000 0400260C
	s_load_b32 s12, s[2:3], 0x1d0                              // 0000000017F4: F4000301 F80001D0
	s_mul_i32 s2, s15, 5                                       // 0000000017FC: 9602850F
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001800: BF870099
	s_ashr_i32 s3, s2, 31                                      // 000000001804: 86039F02
	v_max3_f32 v0, s5, s4, v0                                  // 000000001808: D61C0000 04000805
	s_lshl_b64 s[2:3], s[2:3], 2                               // 000000001810: 84828202
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 000000001814: BF8700A9
	s_add_u32 s2, s0, s2                                       // 000000001818: 80020200
	s_addc_u32 s3, s1, s3                                      // 00000000181C: 82030301
	v_max3_f32 v0, s7, s6, v0                                  // 000000001820: D61C0000 04000C07
	s_ashr_i32 s15, s14, 31                                    // 000000001828: 860F9F0E
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 00000000182C: BF870099
	s_lshl_b64 s[0:1], s[14:15], 2                             // 000000001830: 8480820E
	v_max3_f32 v0, s8, s13, v0                                 // 000000001834: D61C0000 04001A08
	s_add_u32 s0, s2, s0                                       // 00000000183C: 80000002
	s_addc_u32 s1, s3, s1                                      // 000000001840: 82010103
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 000000001844: BF8700A1
	v_max3_f32 v0, s10, s9, v0                                 // 000000001848: D61C0000 0400120A
	s_waitcnt lgkmcnt(0)                                       // 000000001850: BF89FC07
	v_max3_f32 v0, s12, s11, v0                                // 000000001854: D61C0000 0400160C
	global_store_b32 v1, v0, s[0:1]                            // 00000000185C: DC6A0000 00000001
	s_nop 0                                                    // 000000001864: BF800000
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)                       // 000000001868: BFB60003
	s_endpgm                                                   // 00000000186C: BFB00000

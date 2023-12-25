
<stdin>:	file format elf64-amdgpu

Disassembly of section .text:

0000000000001700 <r_3_4_5_3>:
	s_clause 0x1                                               // 000000001700: BF850001
	s_load_b128 s[4:7], s[0:1], null                           // 000000001704: F4080100 F8000000
	s_load_b64 s[0:1], s[0:1], 0x10                            // 00000000170C: F4040000 F8000010
	s_mul_i32 s8, s14, 3                                       // 000000001714: 9608830E
	s_mul_i32 s10, s15, 15                                     // 000000001718: 960A8F0F
	s_ashr_i32 s9, s8, 31                                      // 00000000171C: 86099F08
	s_mov_b32 s2, s13                                          // 000000001720: BE82000D
	s_lshl_b64 s[8:9], s[8:9], 2                               // 000000001724: 84888208
	v_mov_b32_e32 v1, 0                                        // 000000001728: 7E020280
	s_waitcnt lgkmcnt(0)                                       // 00000000172C: BF89FC07
	s_add_u32 s6, s6, s8                                       // 000000001730: 80060806
	s_addc_u32 s7, s7, s9                                      // 000000001734: 82070907
	s_ashr_i32 s11, s10, 31                                    // 000000001738: 860B9F0A
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 00000000173C: BF870499
	s_lshl_b64 s[8:9], s[10:11], 2                             // 000000001740: 8488820A
	s_add_u32 s8, s0, s8                                       // 000000001744: 80080800
	s_addc_u32 s9, s1, s9                                      // 000000001748: 82090901
	s_ashr_i32 s3, s13, 31                                     // 00000000174C: 86039F0D
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000001750: BF870499
	s_lshl_b64 s[0:1], s[2:3], 2                               // 000000001754: 84808202
	s_add_u32 s2, s8, s0                                       // 000000001758: 80020008
	s_addc_u32 s3, s9, s1                                      // 00000000175C: 82030109
	s_load_b64 s[8:9], s[6:7], null                            // 000000001760: F4040203 F8000000
	s_clause 0x1                                               // 000000001768: BF850001
	s_load_b32 s10, s[2:3], null                               // 00000000176C: F4000281 F8000000
	s_load_b32 s11, s[2:3], 0x14                               // 000000001774: F40002C1 F8000014
	s_load_b32 s12, s[6:7], 0x8                                // 00000000177C: F4000303 F8000008
	s_load_b32 s13, s[2:3], 0x28                               // 000000001784: F4000341 F8000028
	s_mul_i32 s2, s15, 20                                      // 00000000178C: 9602940F
	s_mul_i32 s6, s14, 5                                       // 000000001790: 9606850E
	s_ashr_i32 s3, s2, 31                                      // 000000001794: 86039F02
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000001798: BF870499
	s_lshl_b64 s[2:3], s[2:3], 2                               // 00000000179C: 84828202
	s_add_u32 s4, s4, s2                                       // 0000000017A0: 80040204
	s_addc_u32 s5, s5, s3                                      // 0000000017A4: 82050305
	s_ashr_i32 s7, s6, 31                                      // 0000000017A8: 86079F06
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 0000000017AC: BF870499
	s_lshl_b64 s[2:3], s[6:7], 2                               // 0000000017B0: 84828206
	s_add_u32 s2, s4, s2                                       // 0000000017B4: 80020204
	s_addc_u32 s3, s5, s3                                      // 0000000017B8: 82030305
	s_add_u32 s0, s2, s0                                       // 0000000017BC: 80000002
	s_addc_u32 s1, s3, s1                                      // 0000000017C0: 82010103
	s_waitcnt lgkmcnt(0)                                       // 0000000017C4: BF89FC07
	v_fma_f32 v0, s8, s10, 0                                   // 0000000017C8: D6130000 02001408
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 0000000017D0: BF870091
	v_fmac_f32_e64 v0, s9, s11                                 // 0000000017D4: D52B0000 00001609
	v_fmac_f32_e64 v0, s12, s13                                // 0000000017DC: D52B0000 00001A0C
	global_store_b32 v1, v0, s[0:1]                            // 0000000017E4: DC6A0000 00000001
	s_nop 0                                                    // 0000000017EC: BF800000
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)                       // 0000000017F0: BFB60003
	s_endpgm                                                   // 0000000017F4: BFB00000

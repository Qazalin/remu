
<stdin>:	file format elf64-amdgpu

Disassembly of section .text:

0000000000001700 <r_3_5_3>:
	s_clause 0x1                                               // 000000001700: BF850001
	s_load_b64 s[8:9], s[0:1], 0x10                            // 000000001704: F4040200 F8000010
	s_load_b128 s[4:7], s[0:1], null                           // 00000000170C: F4080100 F8000000
	s_mul_i32 s0, s15, 15                                      // 000000001714: 96008F0F
	s_mov_b32 s2, s15                                          // 000000001718: BE82000F
	s_ashr_i32 s1, s0, 31                                      // 00000000171C: 86019F00
	s_mul_i32 s2, s2, 5                                        // 000000001720: 96028502
	s_lshl_b64 s[0:1], s[0:1], 2                               // 000000001724: 84808200
	v_mov_b32_e32 v1, 0                                        // 000000001728: 7E020280
	s_waitcnt lgkmcnt(0)                                       // 00000000172C: BF89FC07
	s_add_u32 s3, s8, s0                                       // 000000001730: 80030008
	s_addc_u32 s9, s9, s1                                      // 000000001734: 82090109
	s_ashr_i32 s15, s14, 31                                    // 000000001738: 860F9F0E
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 00000000173C: BF870499
	s_lshl_b64 s[0:1], s[14:15], 2                             // 000000001740: 8480820E
	s_add_u32 s8, s3, s0                                       // 000000001744: 80080003
	s_addc_u32 s9, s9, s1                                      // 000000001748: 82090109
	s_load_b64 s[10:11], s[6:7], null                          // 00000000174C: F4040283 F8000000
	s_clause 0x1                                               // 000000001754: BF850001
	s_load_b32 s3, s[8:9], null                                // 000000001758: F40000C4 F8000000
	s_load_b32 s12, s[8:9], 0x14                               // 000000001760: F4000304 F8000014
	s_load_b32 s6, s[6:7], 0x8                                 // 000000001768: F4000183 F8000008
	s_load_b32 s7, s[8:9], 0x28                                // 000000001770: F40001C4 F8000028
	s_waitcnt lgkmcnt(0)                                       // 000000001778: BF89FC07
	v_fma_f32 v0, s10, s3, 0                                   // 00000000177C: D6130000 0200060A
	s_ashr_i32 s3, s2, 31                                      // 000000001784: 86039F02
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001788: BF870099
	s_lshl_b64 s[2:3], s[2:3], 2                               // 00000000178C: 84828202
	v_fmac_f32_e64 v0, s11, s12                                // 000000001790: D52B0000 0000180B
	s_add_u32 s2, s4, s2                                       // 000000001798: 80020204
	s_addc_u32 s3, s5, s3                                      // 00000000179C: 82030305
	s_add_u32 s0, s2, s0                                       // 0000000017A0: 80000002
	s_addc_u32 s1, s3, s1                                      // 0000000017A4: 82010103
	v_fmac_f32_e64 v0, s6, s7                                  // 0000000017A8: D52B0000 00000E06
	global_store_b32 v1, v0, s[0:1]                            // 0000000017B0: DC6A0000 00000001
	s_nop 0                                                    // 0000000017B8: BF800000
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)                       // 0000000017BC: BFB60003
	s_endpgm                                                   // 0000000017C0: BFB00000
